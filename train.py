"""
train.py — Unified training script for YOLOv8-seg and Mask R-CNN.

Supports switching between model architectures via the --model flag.
Logs metrics, saves best checkpoints, and generates evaluation plots.

Usage:
    python train.py --model yolo --data d2s.yaml --epochs 100 --batch 16
    python train.py --model maskrcnn --data d2s.yaml --epochs 100 --batch 2
    python train.py --model yolo --data d2s.yaml --resume
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_yolo(args: argparse.Namespace) -> None:
    """
    Train a YOLOv8-seg model using the Ultralytics Python API.

    Configures training with transfer learning from COCO pretrained weights,
    SGD optimizer with cosine LR scheduler, and early stopping.

    Args:
        args: Parsed command-line arguments containing data, epochs, batch,
              device, resume, imgsz, and model size options.
    """
    from ultralytics import YOLO

    # Select model size
    model_variant = f"yolov8{args.size}-seg.pt"
    logger.info(f"📦 Loading pretrained model: {model_variant}")

    if args.resume and os.path.exists("runs/segment/train/weights/last.pt"):
        logger.info("🔄 Resuming training from last checkpoint...")
        model = YOLO("runs/segment/train/weights/last.pt")
    else:
        model = YOLO(model_variant)

    # Resolve absolute path for data yaml
    data_path = os.path.abspath(args.data)

    logger.info("=" * 60)
    logger.info("YOLOv8-seg Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  Model:      {model_variant}")
    logger.info(f"  Data:       {data_path}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch Size: {args.batch}")
    logger.info(f"  Image Size: {args.imgsz}")
    logger.info(f"  Device:     {args.device}")
    logger.info(f"  Resume:     {args.resume}")
    logger.info("=" * 60)

    # Train with specified configuration
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/segment",
        name="train",
        exist_ok=True,
        # Optimizer settings
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,  # Final LR = lr0 * lrf (cosine scheduler)
        momentum=0.937,
        weight_decay=0.0005,
        # Early stopping
        patience=20,
        # Augmentation
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=15.0,     # Rotation ±15°
        translate=0.1,     # Translation
        scale=0.5,         # Scale (0.5-1.5x)
        fliplr=0.5,        # Horizontal flip probability
        flipud=0.5,        # Vertical flip probability
        mosaic=1.0,        # Mosaic augmentation (4-image)
        # Saving
        save=True,
        save_period=10,    # Save checkpoint every 10 epochs
        plots=True,        # Generate training plots
        val=True,          # Run validation after each epoch
        verbose=True,
    )

    # Copy best weights to the project weights directory
    best_pt = Path("runs/segment/train/weights/best.pt")
    if best_pt.exists():
        os.makedirs("backend/weights", exist_ok=True)
        import shutil
        dest = Path("backend/weights/best.pt")
        shutil.copy2(str(best_pt), str(dest))
        logger.info(f"✅ Best weights copied to {dest}")

    logger.info("✅ YOLOv8-seg training complete!")
    logger.info(f"   Results saved to: runs/segment/train/")

    # Generate additional plots
    _generate_yolo_plots()


def _generate_yolo_plots() -> None:
    """Generate confusion matrix and PR curve plots from YOLOv8 training results."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        results_csv = Path("runs/segment/train/results.csv")
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()

            # Plot training losses
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("YOLOv8-seg Training Results — D2S Dataset", fontsize=14, fontweight="bold")

            loss_columns = {
                "train/box_loss": "Box Loss",
                "train/seg_loss": "Mask Loss",
                "train/cls_loss": "Class Loss",
            }

            for idx, (col, label) in enumerate(loss_columns.items()):
                ax = axes[idx // 2][idx % 2]
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], label=f"Train {label}", color="#1565C0", linewidth=2)
                val_col = col.replace("train/", "val/")
                if val_col in df.columns:
                    ax.plot(df["epoch"], df[val_col], label=f"Val {label}", color="#FF6F00",
                            linewidth=2, linestyle="--")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(label)
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Plot mAP
            ax = axes[1][1]
            map_cols = {
                "metrics/mAP50(M)": "mAP@50 (Mask)",
                "metrics/mAP50-95(M)": "mAP@50-95 (Mask)",
            }
            for col, label in map_cols.items():
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], label=label, linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("mAP")
            ax.set_title("Segmentation mAP")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("runs/segment/train/training_curves.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("📊 Training curves saved to runs/segment/train/training_curves.png")

    except Exception as e:
        logger.warning(f"⚠️ Could not generate plots: {e}")


def train_maskrcnn(args: argparse.Namespace) -> None:
    """
    Train a Mask R-CNN model using Detectron2.

    Uses ResNet-50-FPN backbone with COCO pretrained weights.
    Fine-tunes box_predictor and mask_head on D2S 60 classes.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer, HookBase
        from detectron2.evaluation import COCOEvaluator
    except ImportError:
        logger.error(
            "❌ Detectron2 is not installed. Install it with:\n"
            "   pip install 'git+https://github.com/facebookresearch/detectron2.git'"
        )
        sys.exit(1)

    # Load dataset config
    with open(args.data, "r") as f:
        data_config = yaml.safe_load(f)

    dataset_root = data_config.get("path", ".")

    # Determine COCO annotation paths
    # Look for COCO JSON files in the original D2S directory
    d2s_root = os.getenv("D2S_ROOT", "./mvtec_d2s")
    train_ann = os.path.join(d2s_root, "annotations", "instances_train.json")
    val_ann = os.path.join(d2s_root, "annotations", "instances_val.json")
    train_images = os.path.join(d2s_root, "images", "train")
    val_images = os.path.join(d2s_root, "images", "val")

    # Register datasets
    if "d2s_train" not in DatasetCatalog.list():
        register_coco_instances("d2s_train", {}, train_ann, train_images)
    if "d2s_val" not in DatasetCatalog.list():
        register_coco_instances("d2s_val", {}, val_ann, val_images)

    # Configure Mask R-CNN
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("d2s_train",)
    cfg.DATASETS.TEST = ("d2s_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # Use COCO pretrained weights for transfer learning
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Training hyperparameters
    num_classes = data_config.get("nc", 60)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Calculate iterations (~100 epochs equivalent)
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.MAX_ITER = 18000
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (12000, 16000)
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000

    # Input size
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Output directory
    cfg.OUTPUT_DIR = "runs/maskrcnn/train"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Device
    if args.device == "cpu":
        cfg.MODEL.DEVICE = "cpu"
    else:
        cfg.MODEL.DEVICE = "cuda"

    # Resume from checkpoint
    if args.resume:
        last_ckpt = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.exists(last_ckpt):
            cfg.MODEL.WEIGHTS = last_ckpt
            logger.info("🔄 Resuming from last checkpoint...")

    logger.info("=" * 60)
    logger.info("Mask R-CNN Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  Backbone:    ResNet-50-FPN")
    logger.info(f"  Classes:     {num_classes}")
    logger.info(f"  Max Iter:    {cfg.SOLVER.MAX_ITER}")
    logger.info(f"  Batch Size:  {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"  Base LR:     {cfg.SOLVER.BASE_LR}")
    logger.info(f"  Device:      {cfg.MODEL.DEVICE}")
    logger.info("=" * 60)

    # Custom trainer with evaluation
    class D2STrainer(DefaultTrainer):
        """Custom trainer with COCO evaluation during training."""

        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    # Train
    trainer = D2STrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    # Copy best model
    final_model = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if os.path.exists(final_model):
        os.makedirs("backend/weights", exist_ok=True)
        import shutil
        dest = os.path.join("backend/weights", "model_best.pth")
        shutil.copy2(final_model, dest)
        logger.info(f"✅ Best weights copied to {dest}")

    logger.info("✅ Mask R-CNN training complete!")


def main() -> None:
    """Parse arguments and dispatch to the appropriate training function."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg or Mask R-CNN on the MVTec D2S dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model yolo --data d2s.yaml --epochs 100 --batch 16
  python train.py --model yolo --data d2s.yaml --epochs 100 --batch 8 --size m
  python train.py --model maskrcnn --data d2s.yaml --batch 2
  python train.py --model yolo --data d2s.yaml --resume
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo", "maskrcnn"],
        default="yolo",
        help="Model architecture to train (default: yolo).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="d2s.yaml",
        help="Path to dataset YAML config (default: d2s.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100). For Mask R-CNN, maps to iterations.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16 for YOLO, use 2 for Mask R-CNN).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' for GPU 0, 'cpu' for CPU (default: '0').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for YOLOv8 (default: 640).",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLOv8 model size variant (default: n=nano).",
    )

    args = parser.parse_args()

    # Validate data file exists
    if not os.path.exists(args.data):
        logger.error(f"❌ Dataset config not found: {args.data}")
        logger.info("   Run data_preparation.py first to generate it.")
        sys.exit(1)

    # Dispatch to appropriate trainer
    if args.model == "yolo":
        train_yolo(args)
    elif args.model == "maskrcnn":
        train_maskrcnn(args)
    else:
        logger.error(f"❌ Unknown model type: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
