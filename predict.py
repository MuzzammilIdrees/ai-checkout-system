"""
predict.py — Single-image inference with visualisation for the AI Checkout System.

Accepts a single image path, runs instance segmentation inference, draws
coloured masks with labels and confidence scores, and saves the output image.

Usage:
    python predict.py --image test.jpg --model yolo --weights best.pt
    python predict.py --image test.jpg --model yolo --weights best.pt --output result.jpg
    python predict.py --image test.jpg --model maskrcnn --weights model_best.pth
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 20 distinct colours (BGR format for OpenCV)
PALETTE_BGR = [
    (34, 87, 255), (243, 150, 33), (80, 175, 76), (7, 193, 255),
    (176, 39, 156), (212, 188, 0), (51, 87, 255), (181, 81, 63),
    (74, 195, 139), (0, 152, 255), (99, 30, 233), (136, 150, 0),
    (72, 85, 121), (59, 235, 255), (183, 58, 103), (54, 67, 244),
    (244, 169, 3), (57, 220, 205), (139, 125, 96), (67, 112, 255),
]

PALETTE_HEX = [
    "#FF5722", "#2196F3", "#4CAF50", "#FFC107", "#9C27B0",
    "#00BCD4", "#FF5733", "#3F51B5", "#8BC34A", "#FF9800",
    "#E91E63", "#009688", "#795548", "#FFEB3B", "#673AB7",
    "#F44336", "#03A9F4", "#CDDC39", "#607D8B", "#FF7043",
]


def predict_yolo(image_path: str, weights: str, conf: float, iou: float, imgsz: int) -> tuple:
    """
    Run YOLOv8-seg inference on a single image.

    Args:
        image_path: Path to the input image.
        weights: Path to YOLOv8-seg weights file.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        imgsz: Input image size.

    Returns:
        Tuple of (original_image, results_object, class_names_list).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    image = cv2.imread(image_path)

    if image is None:
        logger.error(f"❌ Failed to read image: {image_path}")
        sys.exit(1)

    start = time.perf_counter()
    results = model(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000

    # Get class names
    class_names = []
    if hasattr(model, "names"):
        if isinstance(model.names, dict):
            class_names = [model.names[i] for i in sorted(model.names.keys())]
        else:
            class_names = list(model.names)

    logger.info(f"⏱️ Inference time: {elapsed:.1f}ms")

    return image, results, class_names


def predict_maskrcnn(image_path: str, weights: str, conf: float) -> tuple:
    """
    Run Mask R-CNN inference on a single image using Detectron2.

    Args:
        image_path: Path to the input image.
        weights: Path to Mask R-CNN weights file.
        conf: Confidence threshold.

    Returns:
        Tuple of (original_image, outputs_dict, class_names_list).
    """
    try:
        import torch
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
    except ImportError:
        logger.error("❌ Detectron2 required for Mask R-CNN. Install it first.")
        sys.exit(1)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"❌ Failed to read image: {image_path}")
        sys.exit(1)

    start = time.perf_counter()
    outputs = predictor(image)
    elapsed = (time.perf_counter() - start) * 1000

    # Load class names
    class_names = [f"class_{i}" for i in range(60)]
    categories_path = Path("categories.json")
    if categories_path.exists():
        import json
        with open(categories_path, "r") as f:
            cats = json.load(f)
        if isinstance(cats, list):
            class_names = [c.get("name", f"class_{i}") for i, c in enumerate(cats)]

    logger.info(f"⏱️ Inference time: {elapsed:.1f}ms")

    return image, outputs, class_names


def draw_yolo_results(image: np.ndarray, results, class_names: list[str]) -> np.ndarray:
    """
    Draw YOLOv8-seg results on the image with coloured masks and labels.

    Args:
        image: Original image (BGR numpy array).
        results: YOLOv8 results object.
        class_names: List of class names.

    Returns:
        Image with drawn masks and labels.
    """
    output = image.copy()
    overlay = image.copy()

    detection_count = 0
    item_counts: dict[str, int] = {}

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes
        masks = result.masks

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().astype(int)

            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            colour_bgr = PALETTE_BGR[class_id % len(PALETTE_BGR)]

            item_counts[class_name] = item_counts.get(class_name, 0) + 1

            # Draw mask polygon
            if masks is not None and i < len(masks) and masks.xy is not None:
                polygon = masks.xy[i]
                if len(polygon) > 0:
                    points = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(overlay, [points], colour_bgr)
                    cv2.polylines(output, [points], True, colour_bgr, 2)

            # Draw label
            label = f"{class_name} {confidence:.0%}"
            x1, y1 = bbox[0], bbox[1]
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour_bgr, -1)
            cv2.putText(output, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            detection_count += 1

    # Blend overlay for semi-transparent masks (opacity 0.45)
    cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)

    # Draw summary box
    _draw_summary(output, detection_count, item_counts)

    return output


def draw_maskrcnn_results(image: np.ndarray, outputs, class_names: list[str]) -> np.ndarray:
    """
    Draw Mask R-CNN results on the image with coloured masks and labels.

    Args:
        image: Original image (BGR numpy array).
        outputs: Detectron2 prediction outputs.
        class_names: List of class names.

    Returns:
        Image with drawn masks and labels.
    """
    output = image.copy()
    overlay = image.copy()

    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        logger.info("📭 No detections found.")
        return output

    boxes = instances.pred_boxes.tensor.numpy().astype(int)
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    masks = instances.pred_masks.numpy()

    detection_count = len(boxes)
    item_counts: dict[str, int] = {}

    for i in range(len(boxes)):
        class_id = int(classes[i])
        confidence = float(scores[i])
        bbox = boxes[i]

        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        colour_bgr = PALETTE_BGR[class_id % len(PALETTE_BGR)]

        item_counts[class_name] = item_counts.get(class_name, 0) + 1

        # Draw binary mask
        binary_mask = masks[i].astype(np.uint8)
        coloured_mask = np.zeros_like(image)
        coloured_mask[binary_mask > 0] = colour_bgr
        overlay = cv2.addWeighted(overlay, 1.0, coloured_mask, 0.45, 0)

        # Draw contour
        contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, colour_bgr, 2)

        # Draw label
        label = f"{class_name} {confidence:.0%}"
        x1, y1 = bbox[0], bbox[1]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour_bgr, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)
    _draw_summary(output, detection_count, item_counts)

    return output


def _draw_summary(image: np.ndarray, count: int, items: dict[str, int]) -> None:
    """
    Draw a summary box on the image showing detection counts.

    Args:
        image: Image to draw on (modified in-place).
        count: Total number of detections.
        items: Dictionary of item_name → quantity.
    """
    h, w = image.shape[:2]
    margin = 15
    line_height = 25
    box_height = 40 + len(items) * line_height
    box_width = 350

    # Semi-transparent background
    sub_img = image[margin: margin + box_height, margin: margin + box_width]
    dark_rect = np.zeros_like(sub_img)
    blended = cv2.addWeighted(sub_img, 0.4, dark_rect, 0.6, 0)
    image[margin: margin + box_height, margin: margin + box_width] = blended

    # Title
    cv2.putText(
        image, f"Detected: {count} items",
        (margin + 10, margin + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    # Item list
    y = margin + 50
    for name, qty in sorted(items.items()):
        cv2.putText(
            image, f"  {name}: x{qty}",
            (margin + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        y += line_height


def main() -> None:
    """Parse arguments and run single-image prediction with visualisation."""
    parser = argparse.ArgumentParser(
        description="Run instance segmentation inference on a single image and save visualisation.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo", "maskrcnn"],
        default="yolo",
        help="Model type (default: yolo).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/segment/train/weights/best.pt",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <input>_predicted.jpg).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for YOLO (default: 640).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        logger.error(f"❌ Image not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.weights):
        logger.error(f"❌ Weights not found: {args.weights}")
        sys.exit(1)

    # Determine output path
    if args.output is None:
        stem = Path(args.image).stem
        args.output = f"{stem}_predicted.jpg"

    logger.info(f"🖼️ Input:   {args.image}")
    logger.info(f"📦 Model:   {args.model}")
    logger.info(f"🎯 Weights: {args.weights}")

    # Run inference and draw results
    if args.model == "yolo":
        image, results, class_names = predict_yolo(
            args.image, args.weights, args.conf, args.iou, args.imgsz
        )
        output_image = draw_yolo_results(image, results, class_names)
    elif args.model == "maskrcnn":
        image, outputs, class_names = predict_maskrcnn(
            args.image, args.weights, args.conf
        )
        output_image = draw_maskrcnn_results(image, outputs, class_names)
    else:
        logger.error(f"❌ Unknown model: {args.model}")
        sys.exit(1)

    # Save output
    cv2.imwrite(args.output, output_image)
    logger.info(f"💾 Output saved to: {args.output}")
    logger.info("✅ Prediction complete!")


if __name__ == "__main__":
    main()
