# Dataset Preparation Guide — MVTec D2S for YOLOv8-seg

## Overview

This guide explains how to convert the MVTec D2S (Densely Segmented Supermarket) dataset from its native COCO JSON annotation format to the YOLO segmentation label format required by Ultralytics YOLOv8-seg.

## Prerequisites

- MVTec D2S dataset downloaded and extracted locally
- Python 3.10+ with the following packages:
  ```bash
  pip install pycocotools opencv-python Pillow PyYAML tqdm numpy
  ```

## Expected Input Directory Structure

Your MVTec D2S dataset should be organized as follows:

```
mvtec_d2s/
├── images/
│   ├── train/              # Single-item images (clean backgrounds)
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── val/                # Multi-item scenes (occlusion, overlapping)
│       ├── 000001.jpg
│       └── ...
├── annotations/
│   ├── instances_train.json    # COCO-format segmentation annotations
│   └── instances_val.json
└── categories.json             # 60 category names and IDs
```

## Step-by-Step Conversion

### Step 1: Verify Dataset

Check that your dataset is complete:

```bash
# Count images
ls mvtec_d2s/images/train/ | wc -l
ls mvtec_d2s/images/val/ | wc -l

# Check annotation files exist
ls mvtec_d2s/annotations/
```

### Step 2: Run the Conversion Script

```bash
python data_preparation.py \
    --d2s-root ./mvtec_d2s \
    --output ./mvtec_d2s_yolo \
    --yaml-ref ./d2s.yaml
```

**Arguments:**
- `--d2s-root`: Path to the MVTec D2S root directory
- `--output`: Output directory for YOLO-formatted dataset
- `--yaml-ref`: Reference YAML for consistent class ordering (uses d2s.yaml in project root)

### Step 3: Verify Output

The conversion creates the following structure:

```
mvtec_d2s_yolo/
├── images/
│   ├── train/          # Copied training images
│   └── val/            # Copied validation images
├── labels/
│   ├── train/          # YOLO .txt label files (one per image)
│   │   ├── 000001.txt
│   │   └── ...
│   └── val/
│       ├── 000001.txt
│       └── ...
└── d2s.yaml            # Ultralytics dataset config
```

### Step 4: Understand the Label Format

Each YOLO segmentation label file contains one line per object instance:

```
class_id x1 y1 x2 y2 x3 y3 x4 y4 ...
```

- `class_id`: Integer (0-59) matching the class index in d2s.yaml
- `x, y`: Normalized polygon coordinates (0.0 to 1.0)

**Example:**
```
0 0.156250 0.333333 0.468750 0.333333 0.468750 0.666667 0.156250 0.666667
5 0.078125 0.083333 0.234375 0.083333 0.234375 0.416667 0.078125 0.416667
```

This means:
- Line 1: Class 0 (juice_auer_cranberry) with a 4-point polygon mask
- Line 2: Class 5 (water_juvina) with a 4-point polygon mask

### Step 5: Verify d2s.yaml

The generated `d2s.yaml` should contain:

```yaml
path: /absolute/path/to/mvtec_d2s_yolo
train: images/train
val: images/val
nc: 60
names:
  0: juice_auer_cranberry
  1: juice_auer_apple
  # ... all 60 classes
  59: fruit_apple
```

## COCO to YOLO Conversion Details

### What Gets Converted

| COCO Field | YOLO Equivalent |
|-----------|----------------|
| `images[].file_name` | Image copied to `images/train/` or `images/val/` |
| `annotations[].category_id` | Mapped to sequential 0-59 class index |
| `annotations[].segmentation` | Polygon coordinates normalized to [0, 1] |
| `categories[]` | Class names in d2s.yaml |

### What Gets Skipped

- RLE-encoded masks (only polygon format is converted)
- Annotations with fewer than 3 polygon points (degenerate)
- Images without any annotations (no label file created)

## Data Augmentation

The following augmentations are configured in `train.py` for YOLOv8-seg training:

| Augmentation | Value | Description |
|-------------|-------|-------------|
| Horizontal Flip | 50% | Random left-right flip |
| Vertical Flip | 50% | Random top-bottom flip |
| Rotation | ±15° | Random rotation |
| Scale | 0.5-1.5× | Random scale |
| HSV-Hue | 0.015 | Color jitter |
| HSV-Saturation | 0.7 | Saturation jitter |
| HSV-Value | 0.4 | Brightness jitter |
| Mosaic | 100% | 4-image mosaic for dense scene simulation |

## Troubleshooting

### "No annotations found for image"
- Check that `instances_train.json` and `instances_val.json` are correct COCO format
- Verify image file names match between annotation JSON and actual files

### "Category ID not found in mapping"
- Ensure `categories.json` or the annotation file has all 60 categories
- Check that `d2s.yaml` has matching class names

### "Image not found"
- Verify the `--d2s-root` path points to the correct directory
- Check that images are in `images/train/` and `images/val/` subdirectories

### Large file sizes
- Original D2S images are high-res (4096×3000). They will be resized to 640×640 during training by YOLOv8
- The conversion script copies images as-is; YOLOv8 handles resizing internally
