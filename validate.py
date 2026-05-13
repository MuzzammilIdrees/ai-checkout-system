"""
validate.py — Full COCO metrics evaluation script for the AI Checkout System.

Loads the best trained checkpoint and runs evaluation on the validation split.
Prints full COCO metrics (AP, AP50, AP75, APs, APm, APl) as well as per-class
AP for all 60 D2S categories.

Usage:
    python validate.py --model yolo --weights runs/segment/train/weights/best.pt --data d2s.yaml
    python validate.py --model maskrcnn --weights runs/maskrcnn/train/model_final.pth --data d2s.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_yolo(args: argparse.Namespace) -> None:
    """
    Validate a YOLOv8-seg model on the D2S validation set.

    Runs full COCO evaluation and prints per-class metrics.

    Args:
        args: Parsed arguments with weights, data, device, and imgsz.
    """
    from ultralytics import YOLO

    if not os.path.exists(args.weights):
        logger.error(f"❌ Weights file not found: {args.weights}")
        sys.exit(1)

    logger.info(f"📦 Loading model from: {args.weights}")
    model = YOLO(args.weights)

    logger.info("=" * 60)
    logger.info("YOLOv8-seg Validation")
    logger.info("=" * 60)
    logger.info(f"  Weights:    {args.weights}")
    logger.info(f"  Data:       {args.data}")
    logger.info(f"  Device:     {args.device}")
    logger.info(f"  Image Size: {args.imgsz}")
    logger.info("=" * 60)

    # Run validation
    results = model.val(
        data=os.path.abspath(args.data),
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        plots=True,
        save_json=True,
        verbose=True,
    )

    # Print summary metrics
    logger.info("")
    logger.info("=" * 60)
    logger.info("COCO Metrics Summary (Segmentation)")
    logger.info("=" * 60)

    if hasattr(results, "seg"):
        seg = results.seg
        logger.info(f"  mAP@50:     {seg.map50:.4f}")
        logger.info(f"  mAP@50-95:  {seg.map:.4f}")
        logger.info(f"  mAP@75:     {seg.map75:.4f}")

        # Per-class AP
        logger.info("")
        logger.info("Per-Class AP (Segmentation):")
        logger.info("-" * 50)

        # Load class names
        with open(args.data, "r") as f:
            data_config = yaml.safe_load(f)

        names = data_config.get("names", {})
        if isinstance(names, dict):
            class_names = [names.get(i, f"class_{i}") for i in range(len(names))]
        else:
            class_names = list(names)

        if hasattr(seg, "ap50") and seg.ap50 is not None:
            for i, ap in enumerate(seg.ap50):
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                logger.info(f"  {name:40s}  AP@50: {ap:.4f}")

    if hasattr(results, "box"):
        box = results.box
        logger.info("")
        logger.info("COCO Metrics Summary (Detection / Boxes)")
        logger.info("-" * 50)
        logger.info(f"  mAP@50:     {box.map50:.4f}")
        logger.info(f"  mAP@50-95:  {box.map:.4f}")
        logger.info(f"  mAP@75:     {box.map75:.4f}")

    logger.info("")
    logger.info("✅ Validation complete!")


def validate_maskrcnn(args: argparse.Namespace) -> None:
    """
    Validate a Mask R-CNN model on the D2S validation set using Detectron2.

    Runs COCO evaluation with full AP metrics.

    Args:
        args: Parsed arguments with weights and data paths.
    """
    try:
        import torch
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultPredictor
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
    except ImportError:
        logger.error("❌ Detectron2 is not installed. Required for Mask R-CNN validation.")
        sys.exit(1)

    if not os.path.exists(args.weights):
        logger.error(f"❌ Weights file not found: {args.weights}")
        sys.exit(1)

    # Load dataset config
    with open(args.data, "r") as f:
        data_config = yaml.safe_load(f)

    num_classes = data_config.get("nc", 60)

    # Register validation dataset
    d2s_root = os.getenv("D2S_ROOT", "./mvtec_d2s")
    val_ann = os.path.join(d2s_root, "annotations", "instances_val.json")
    val_images = os.path.join(d2s_root, "images", "val")

    if "d2s_val" not in DatasetCatalog.list():
        register_coco_instances("d2s_val", {}, val_ann, val_images)

    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35
    cfg.MODEL.DEVICE = args.device if args.device != "0" else "cuda"
    cfg.DATASETS.TEST = ("d2s_val",)

    logger.info("=" * 60)
    logger.info("Mask R-CNN Validation")
    logger.info("=" * 60)
    logger.info(f"  Weights:  {args.weights}")
    logger.info(f"  Classes:  {num_classes}")
    logger.info(f"  Device:   {cfg.MODEL.DEVICE}")
    logger.info("=" * 60)

    # Run evaluation
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("d2s_val", output_dir="runs/maskrcnn/val")
    val_loader = build_detection_test_loader(cfg, "d2s_val")

    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Print results
    logger.info("")
    logger.info("COCO Evaluation Results:")
    logger.info("-" * 50)

    if "segm" in results:
        segm = results["segm"]
        logger.info("Segmentation Metrics:")
        logger.info(f"  AP:       {segm.get('AP', 0):.4f}")
        logger.info(f"  AP50:     {segm.get('AP50', 0):.4f}")
        logger.info(f"  AP75:     {segm.get('AP75', 0):.4f}")
        logger.info(f"  APs:      {segm.get('APs', 0):.4f}")
        logger.info(f"  APm:      {segm.get('APm', 0):.4f}")
        logger.info(f"  APl:      {segm.get('APl', 0):.4f}")

    if "bbox" in results:
        bbox = results["bbox"]
        logger.info("Detection (Box) Metrics:")
        logger.info(f"  AP:       {bbox.get('AP', 0):.4f}")
        logger.info(f"  AP50:     {bbox.get('AP50', 0):.4f}")
        logger.info(f"  AP75:     {bbox.get('AP75', 0):.4f}")

    logger.info("")
    logger.info("✅ Mask R-CNN validation complete!")


def main() -> None:
    """Parse arguments and run validation for the selected model."""
    parser = argparse.ArgumentParser(
        description="Validate trained segmentation model on D2S validation set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo", "maskrcnn"],
        default="yolo",
        help="Model type to validate (default: yolo).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/segment/train/weights/best.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="d2s.yaml",
        help="Path to dataset YAML config (default: d2s.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' for GPU 0, 'cpu' for CPU.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for validation (default: 640).",
    )

    args = parser.parse_args()

    if args.model == "yolo":
        validate_yolo(args)
    elif args.model == "maskrcnn":
        validate_maskrcnn(args)


if __name__ == "__main__":
    main()
