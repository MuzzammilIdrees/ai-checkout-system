"""
data_preparation.py — COCO JSON → YOLO format converter for MVTec D2S.

Converts the MVTec D2S dataset from COCO JSON annotation format to the YOLO
segmentation label format (.txt files with normalized polygon coordinates).
Also generates the d2s.yaml dataset configuration file.

Handles the actual D2S dataset structure:
  - Images in:      d2s_images_v1/images/  (flat directory, all splits mixed)
  - Annotations in: d2s_annotations_v1.1/annotations/D2S_training.json
                    d2s_annotations_v1.1/annotations/D2S_validation.json

Usage:
    python data_preparation.py
    python data_preparation.py --images-dir ./d2s_images_v1/images --annotations-dir ./d2s_annotations_v1.1/annotations --output ./mvtec_d2s_yolo
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_coco_annotations(annotation_path: str) -> dict[str, Any]:
    """
    Load COCO-format JSON annotation file.

    Args:
        annotation_path: Path to the COCO JSON annotation file.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        f"📂 Loaded annotations from {annotation_path}: "
        f"{len(data.get('images', []))} images, "
        f"{len(data.get('annotations', []))} annotations, "
        f"{len(data.get('categories', []))} categories"
    )
    return data


def build_category_mapping(coco_data: dict[str, Any]) -> dict[int, int]:
    """
    Build a mapping from COCO category IDs to sequential YOLO class indices.

    D2S COCO category IDs are 1-60, YOLO needs 0-59.

    Args:
        coco_data: Parsed COCO JSON dictionary.

    Returns:
        Dictionary mapping COCO category_id → YOLO class_index (0-based).
    """
    coco_categories = coco_data.get("categories", [])

    # Sort by COCO ID for deterministic ordering
    sorted_cats = sorted(coco_categories, key=lambda c: c["id"])

    cat_id_to_yolo_idx: dict[int, int] = {}
    for yolo_idx, cat in enumerate(sorted_cats):
        cat_id_to_yolo_idx[cat["id"]] = yolo_idx

    logger.info(f"📋 Category mapping built: {len(cat_id_to_yolo_idx)} categories (COCO IDs → YOLO 0-{len(cat_id_to_yolo_idx)-1})")
    return cat_id_to_yolo_idx


def coco_polygon_to_yolo(
    segmentation: list[list[float]], img_width: int, img_height: int
) -> list[list[float]]:
    """
    Convert COCO polygon segmentation to YOLO normalized format.

    COCO format: list of polygons, each polygon = [x1, y1, x2, y2, ...]
    YOLO format: same polygon but with coordinates normalized to [0, 1]

    Args:
        segmentation: COCO-style segmentation polygons.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of normalized polygon coordinate lists.
    """
    normalized_polygons: list[list[float]] = []

    for polygon in segmentation:
        if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
            continue

        normalized: list[float] = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            # Clamp to [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            normalized.extend([x, y])

        normalized_polygons.append(normalized)

    return normalized_polygons


def convert_split(
    coco_data: dict[str, Any],
    cat_mapping: dict[int, int],
    images_src_dir: str,
    images_dst_dir: str,
    labels_dst_dir: str,
    split_name: str,
) -> int:
    """
    Convert a single dataset split (train or val) from COCO to YOLO format.

    For each image:
    1. Copies the image to the output directory.
    2. Creates a .txt label file with YOLO segmentation annotations.

    Args:
        coco_data: Parsed COCO JSON for this split.
        cat_mapping: COCO category_id → YOLO class_index mapping.
        images_src_dir: Source directory containing original images.
        images_dst_dir: Destination directory for images.
        labels_dst_dir: Destination directory for label .txt files.
        split_name: Name of the split ('train' or 'val') for logging.

    Returns:
        Number of images processed.
    """
    os.makedirs(images_dst_dir, exist_ok=True)
    os.makedirs(labels_dst_dir, exist_ok=True)

    # Build image_id → image_info lookup
    images_info = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    processed = 0
    skipped = 0

    for img_id, img_info in tqdm(images_info.items(), desc=f"Converting {split_name}"):
        file_name = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Source image path — try multiple locations
        base_name = os.path.basename(file_name)
        src_path = os.path.join(images_src_dir, file_name)

        if not os.path.exists(src_path):
            # Try just the basename (flat directory)
            src_path = os.path.join(images_src_dir, base_name)

        if not os.path.exists(src_path):
            logger.warning(f"⚠️ Image not found: {base_name}")
            skipped += 1
            continue

        # Copy image to destination
        dst_img_path = os.path.join(images_dst_dir, base_name)
        if not os.path.exists(dst_img_path):
            shutil.copy2(src_path, dst_img_path)

        # Create label file
        label_name = os.path.splitext(base_name)[0] + ".txt"
        label_path = os.path.join(labels_dst_dir, label_name)

        annotations = annotations_by_image.get(img_id, [])

        with open(label_path, "w") as f:
            for ann in annotations:
                cat_id = ann["category_id"]
                if cat_id not in cat_mapping:
                    continue

                yolo_class = cat_mapping[cat_id]
                segmentation = ann.get("segmentation", [])

                if not segmentation or isinstance(segmentation, dict):
                    # Skip RLE encoded masks — only handle polygon format
                    continue

                normalized_polys = coco_polygon_to_yolo(segmentation, img_width, img_height)

                for poly in normalized_polys:
                    # YOLO seg format: class_idx x1 y1 x2 y2 x3 y3 ...
                    coords_str = " ".join(f"{c:.6f}" for c in poly)
                    f.write(f"{yolo_class} {coords_str}\n")

        processed += 1

    logger.info(f"✅ {split_name}: processed {processed} images, skipped {skipped}")
    return processed


def generate_yaml_config(
    output_dir: str, coco_data: dict[str, Any], cat_mapping: dict[int, int]
) -> str:
    """
    Generate the d2s.yaml dataset configuration file for Ultralytics training.

    Args:
        output_dir: Root output directory for the YOLO dataset.
        coco_data: COCO JSON data (to extract category names).
        cat_mapping: COCO category_id → YOLO class_index mapping.

    Returns:
        Path to the generated YAML file.
    """
    # Build names dict from COCO categories
    coco_cats = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    names = {}
    for coco_id, yolo_idx in cat_mapping.items():
        if coco_id in coco_cats:
            names[yolo_idx] = coco_cats[coco_id]

    config = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    yaml_path = os.path.join(output_dir, "d2s.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"📄 Dataset YAML written to {yaml_path}")
    return yaml_path


def main() -> None:
    """
    Main entry point for the data preparation pipeline.

    Parses command-line arguments and runs the full COCO → YOLO conversion
    for both train and validation splits.
    """
    parser = argparse.ArgumentParser(
        description="Convert MVTec D2S from COCO JSON to YOLO segmentation format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python data_preparation.py
  python data_preparation.py --images-dir ./d2s_images_v1/images --annotations-dir ./d2s_annotations_v1.1/annotations
        """,
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./d2s_images_v1/images",
        help="Directory containing D2S images (default: ./d2s_images_v1/images).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="./d2s_annotations_v1.1/annotations",
        help="Directory containing D2S COCO JSON annotations (default: ./d2s_annotations_v1.1/annotations).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./mvtec_d2s_yolo",
        help="Output directory for the converted YOLO dataset (default: ./mvtec_d2s_yolo).",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default="D2S_training.json",
        help="Training annotation JSON filename (default: D2S_training.json).",
    )
    parser.add_argument(
        "--val-json",
        type=str,
        default="D2S_validation.json",
        help="Validation annotation JSON filename (default: D2S_validation.json).",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output)

    # Validate input paths
    if not images_dir.exists():
        logger.error(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)

    train_ann = annotations_dir / args.train_json
    val_ann = annotations_dir / args.val_json

    if not train_ann.exists():
        logger.error(f"❌ Training annotations not found: {train_ann}")
        sys.exit(1)

    if not val_ann.exists():
        logger.error(f"❌ Validation annotations not found: {val_ann}")
        sys.exit(1)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MVTec D2S → YOLO Segmentation Format Converter")
    logger.info("=" * 60)
    logger.info(f"📁 Images:      {images_dir}")
    logger.info(f"📁 Annotations: {annotations_dir}")
    logger.info(f"📁 Output:      {output_dir}")
    logger.info("")

    # === Process Training Split ===
    logger.info("━━━ Processing Training Split ━━━")
    train_data = load_coco_annotations(str(train_ann))
    cat_mapping = build_category_mapping(train_data)

    convert_split(
        coco_data=train_data,
        cat_mapping=cat_mapping,
        images_src_dir=str(images_dir),
        images_dst_dir=str(output_dir / "images" / "train"),
        labels_dst_dir=str(output_dir / "labels" / "train"),
        split_name="train",
    )

    # === Process Validation Split ===
    logger.info("━━━ Processing Validation Split ━━━")
    val_data = load_coco_annotations(str(val_ann))

    convert_split(
        coco_data=val_data,
        cat_mapping=cat_mapping,
        images_src_dir=str(images_dir),
        images_dst_dir=str(output_dir / "images" / "val"),
        labels_dst_dir=str(output_dir / "labels" / "val"),
        split_name="val",
    )

    # === Generate YAML config ===
    yaml_path = generate_yaml_config(str(output_dir), train_data, cat_mapping)

    # Also copy to the project root
    project_yaml = Path("d2s.yaml")
    shutil.copy2(yaml_path, str(project_yaml))
    logger.info(f"📄 Copied dataset YAML to project root: {project_yaml}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Data preparation complete!")
    logger.info(f"   Dataset ready at: {output_dir}")
    logger.info(f"   YAML config at:   {yaml_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"   python train.py --model yolo --data {yaml_path} --epochs 5")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
