"""
models/inference.py — Singleton model inference engine for the AI Checkout System.

Supports YOLOv8-seg (primary) and Mask R-CNN (alternative) via MODEL_TYPE env var.
The model is loaded once at application startup via the FastAPI lifespan event.
Thread-safe inference with asyncio locks for concurrent request handling.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectionResultData:
    """Raw detection result from model inference.

    Attributes:
        class_id: Numeric class index from the model.
        class_name: Human-readable category name.
        confidence: Detection confidence score (0-1).
        bbox: Bounding box as [x1, y1, x2, y2].
        mask_polygon: Segmentation polygon as list of [x, y] points.
        mask_area: Area of the mask in pixels.
    """

    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=list)
    mask_polygon: list[list[float]] = field(default_factory=list)
    mask_area: float = 0.0


class InferenceEngine:
    """
    Singleton inference engine that loads and runs the segmentation model.

    The engine supports two model backends:
    - 'yolo': YOLOv8-seg via the Ultralytics library
    - 'maskrcnn': Mask R-CNN via Detectron2

    The model type is determined by the MODEL_TYPE environment variable.
    Model weights are loaded once and kept in memory for zero-overhead inference.
    """

    _instance: Optional["InferenceEngine"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "InferenceEngine":
        """Ensure only one instance of InferenceEngine exists (singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes (only on first creation)."""
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.model_type: str = os.getenv("MODEL_TYPE", "yolo").lower()
        self.weights_path: str = os.getenv("MODEL_WEIGHTS_PATH", "weights/best.pt")
        self.conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.35"))
        self.nms_iou_threshold: float = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))
        self.class_names: list[str] = []
        self.is_loaded: bool = False
        self._inference_lock = asyncio.Lock()
        self._last_processed_image: Optional[np.ndarray] = None

    def load_model(self) -> None:
        """
        Load the segmentation model into memory.

        Reads MODEL_TYPE from environment to decide which backend to use.
        Logs an error and sets is_loaded=False if weights file is missing.
        """
        weights = Path(self.weights_path)

        if not weights.exists():
            logger.error(
                f"❌ Model weights not found at '{self.weights_path}'. "
                f"Inference will return 503. Train the model first and place "
                f"weights in the 'weights/' directory."
            )
            self.is_loaded = False
            return

        try:
            if self.model_type == "yolo":
                self._load_yolo(str(weights))
            elif self.model_type == "maskrcnn":
                self._load_maskrcnn(str(weights))
            else:
                logger.error(f"❌ Unknown MODEL_TYPE: '{self.model_type}'. Use 'yolo' or 'maskrcnn'.")
                self.is_loaded = False
                return

            self.is_loaded = True
            logger.info(
                f"✅ Model loaded successfully: type={self.model_type}, "
                f"weights={self.weights_path}, classes={len(self.class_names)}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False

    def _load_yolo(self, weights_path: str) -> None:
        """
        Load a YOLOv8-seg model using the Ultralytics library.

        Args:
            weights_path: Path to the .pt weights file.
        """
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        # Extract class names from the model
        if hasattr(self.model, "names"):
            if isinstance(self.model.names, dict):
                self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
            else:
                self.class_names = list(self.model.names)
        logger.info(f"📦 YOLOv8-seg loaded with {len(self.class_names)} classes.")

    def _load_maskrcnn(self, weights_path: str) -> None:
        """
        Load a Mask R-CNN model using Detectron2.

        Args:
            weights_path: Path to the .pth weights file.
        """
        try:
            import torch
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor

            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60  # D2S categories
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = DefaultPredictor(cfg)
            self._cfg = cfg

            # Load class names from a categories file if available
            categories_path = Path("categories.json")
            if categories_path.exists():
                import json

                with open(categories_path, "r") as f:
                    cats = json.load(f)
                if isinstance(cats, list):
                    self.class_names = [c.get("name", f"class_{i}") for i, c in enumerate(cats)]
                elif isinstance(cats, dict):
                    self.class_names = [cats.get(str(i), f"class_{i}") for i in range(60)]
            else:
                self.class_names = [f"class_{i}" for i in range(60)]

            logger.info(f"📦 Mask R-CNN loaded with {len(self.class_names)} classes.")
        except ImportError:
            logger.error(
                "❌ Detectron2 is not installed. Install it for Mask R-CNN support: "
                "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )
            raise

    async def run(self, image_bytes: bytes) -> list[DetectionResultData]:
        """
        Run instance segmentation inference on the given image bytes.

        This method is thread-safe — concurrent calls are serialised via an
        asyncio lock to prevent race conditions.

        Args:
            image_bytes: Raw image bytes (JPEG or PNG).

        Returns:
            List of DetectionResultData objects, one per detected instance.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot run inference.")

        async with self._inference_lock:
            start_time = time.perf_counter()

            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image bytes. Ensure valid JPEG/PNG data.")

            # Auto-resize if longer side exceeds 4096px
            h, w = image.shape[:2]
            max_side = 4096
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                logger.info(f"🔄 Image resized from {w}x{h} to {image.shape[1]}x{image.shape[0]}")

            # Run model-specific inference
            if self.model_type == "yolo":
                detections = self._run_yolo(image)
            elif self.model_type == "maskrcnn":
                detections = self._run_maskrcnn(image)
            else:
                detections = []

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"🔍 Inference complete: {len(detections)} detections in {elapsed_ms:.1f}ms"
            )

            # Store last processed image with masks drawn (for debug endpoint)
            self._last_processed_image = self._draw_masks_on_image(image.copy(), detections)

            return detections

    def _run_yolo(self, image: np.ndarray) -> list[DetectionResultData]:
        """
        Run YOLOv8-seg inference on a decoded image.

        Args:
            image: BGR numpy array.

        Returns:
            List of DetectionResultData.
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.nms_iou_threshold,
            verbose=False,
        )

        detections: list[DetectionResultData] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            masks = result.masks

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].tolist()

                # Extract mask polygon
                mask_polygon: list[list[float]] = []
                mask_area: float = 0.0

                if masks is not None and i < len(masks):
                    # Get the mask polygon coordinates
                    if masks.xy is not None and i < len(masks.xy):
                        polygon_points = masks.xy[i]
                        if len(polygon_points) > 0:
                            mask_polygon = [[float(p[0]), float(p[1])] for p in polygon_points]
                            # Calculate mask area using the binary mask
                            if masks.data is not None and i < len(masks.data):
                                binary_mask = masks.data[i].cpu().numpy()
                                mask_area = float(np.sum(binary_mask > 0.5))

                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                detections.append(
                    DetectionResultData(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        mask_polygon=mask_polygon,
                        mask_area=mask_area,
                    )
                )

        return detections

    def _run_maskrcnn(self, image: np.ndarray) -> list[DetectionResultData]:
        """
        Run Mask R-CNN inference on a decoded image.

        Args:
            image: BGR numpy array.

        Returns:
            List of DetectionResultData.
        """
        import torch

        outputs = self.model(image)
        instances = outputs["instances"]

        if len(instances) == 0:
            return []

        # Move to CPU for processing
        instances = instances.to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy()  # Binary masks [N, H, W]

        detections: list[DetectionResultData] = []

        for i in range(len(boxes)):
            class_id = int(classes[i])
            confidence = float(scores[i])
            bbox = boxes[i].tolist()

            # Convert binary mask to polygon
            binary_mask = masks[i].astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask_polygon: list[list[float]] = []
            if contours:
                # Take the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Simplify the contour to reduce points
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                mask_polygon = [[float(p[0][0]), float(p[0][1])] for p in approx]

            mask_area = float(np.sum(binary_mask > 127))

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

            detections.append(
                DetectionResultData(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    mask_polygon=mask_polygon,
                    mask_area=mask_area,
                )
            )

        return detections

    def _draw_masks_on_image(
        self, image: np.ndarray, detections: list[DetectionResultData]
    ) -> np.ndarray:
        """
        Draw coloured segmentation masks and labels on the image for debug display.

        Args:
            image: BGR numpy array to draw on (will be modified in-place).
            detections: List of detection results with mask polygons.

        Returns:
            Image with masks drawn.
        """
        # 20 distinct colours for cycling by class_id
        palette = [
            (255, 87, 34), (33, 150, 243), (76, 175, 80), (255, 193, 7),
            (156, 39, 176), (0, 188, 212), (255, 87, 51), (63, 81, 181),
            (139, 195, 74), (255, 152, 0), (233, 30, 99), (0, 150, 136),
            (121, 85, 72), (255, 235, 59), (103, 58, 183), (244, 67, 54),
            (3, 169, 244), (205, 220, 57), (96, 125, 139), (255, 112, 67),
        ]

        overlay = image.copy()

        for det in detections:
            colour = palette[det.class_id % len(palette)]

            if det.mask_polygon and len(det.mask_polygon) >= 3:
                points = np.array(det.mask_polygon, dtype=np.int32)
                # Draw filled polygon (semi-transparent)
                cv2.fillPoly(overlay, [points], colour)
                # Draw polygon border
                cv2.polylines(image, [points], True, colour, 2)

            # Draw label
            label = f"{det.class_name} {det.confidence:.0%}"
            x1, y1 = int(det.bbox[0]), int(det.bbox[1])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend overlay for semi-transparent masks
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        return image

    def get_last_processed_image(self) -> Optional[np.ndarray]:
        """
        Return the last processed image with masks drawn, for the debug endpoint.

        Returns:
            BGR numpy array or None if no image has been processed yet.
        """
        return self._last_processed_image

    def get_image_dimensions(self, image_bytes: bytes) -> tuple[int, int]:
        """
        Get the dimensions of an image from its raw bytes.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Tuple of (width, height).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return (0, 0)
        h, w = image.shape[:2]
        return (w, h)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        cls._instance = None
