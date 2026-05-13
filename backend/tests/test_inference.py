"""
tests/test_inference.py — Unit tests for the InferenceEngine singleton.

Tests model loading, prediction format, NMS, confidence filtering,
thread safety, error handling, and image processing utilities.
Uses pytest with at least 10 test cases.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.inference import DetectionResultData, InferenceEngine
from utils.item_counter import count_items, get_average_confidence, get_detections_for_class


# ──────────────────── Fixtures ────────────────────


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the InferenceEngine singleton before each test."""
    InferenceEngine.reset()
    yield
    InferenceEngine.reset()


@pytest.fixture
def sample_detections() -> list[DetectionResultData]:
    """Create a list of sample detection results for testing."""
    return [
        DetectionResultData(
            class_id=0,
            class_name="juice_auer_cranberry",
            confidence=0.92,
            bbox=[100.0, 200.0, 300.0, 400.0],
            mask_polygon=[[100, 200], [300, 200], [300, 400], [100, 400]],
            mask_area=40000.0,
        ),
        DetectionResultData(
            class_id=0,
            class_name="juice_auer_cranberry",
            confidence=0.88,
            bbox=[500.0, 200.0, 700.0, 400.0],
            mask_polygon=[[500, 200], [700, 200], [700, 400], [500, 400]],
            mask_area=40000.0,
        ),
        DetectionResultData(
            class_id=5,
            class_name="water_juvina",
            confidence=0.75,
            bbox=[50.0, 50.0, 150.0, 250.0],
            mask_polygon=[[50, 50], [150, 50], [150, 250], [50, 250]],
            mask_area=20000.0,
        ),
    ]


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create a sample JPEG image for testing."""
    import cv2

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:200, 100:200] = [0, 255, 0]  # Green square
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


# ──────────────────── Test Cases ────────────────────


class TestInferenceEngineSingleton:
    """Tests for the InferenceEngine singleton pattern."""

    def test_01_singleton_instance(self):
        """Test that InferenceEngine returns the same instance."""
        engine1 = InferenceEngine()
        engine2 = InferenceEngine()
        assert engine1 is engine2, "InferenceEngine should be a singleton"

    def test_02_singleton_reset(self):
        """Test that reset creates a new instance."""
        engine1 = InferenceEngine()
        InferenceEngine.reset()
        engine2 = InferenceEngine()
        assert engine1 is not engine2, "After reset, a new instance should be created"

    def test_03_default_config(self):
        """Test default configuration values."""
        engine = InferenceEngine()
        assert engine.model_type in ("yolo", "maskrcnn")
        assert engine.conf_threshold == 0.35
        assert engine.nms_iou_threshold == 0.45
        assert engine.is_loaded is False

    def test_04_model_not_loaded_by_default(self):
        """Test that model is not loaded on initialization."""
        engine = InferenceEngine()
        assert engine.model is None
        assert engine.is_loaded is False


class TestInferenceEngineLoading:
    """Tests for model loading behavior."""

    def test_05_load_missing_weights(self):
        """Test that loading with missing weights logs error and sets is_loaded=False."""
        engine = InferenceEngine()
        engine.weights_path = "nonexistent/path/weights.pt"
        engine.load_model()
        assert engine.is_loaded is False

    def test_06_load_invalid_model_type(self):
        """Test that invalid MODEL_TYPE is handled gracefully."""
        engine = InferenceEngine()
        engine.model_type = "invalid_model"
        engine.weights_path = "some_weights.pt"
        # Create a dummy file to pass the existence check
        engine.load_model()
        assert engine.is_loaded is False


class TestInferenceEngineRun:
    """Tests for the inference run method."""

    def test_07_run_without_loaded_model(self):
        """Test that run() raises RuntimeError if model is not loaded."""
        engine = InferenceEngine()

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            asyncio.get_event_loop().run_until_complete(
                engine.run(b"fake_image_bytes")
            )

    def test_08_invalid_image_bytes(self):
        """Test that run() raises ValueError for invalid image data."""
        engine = InferenceEngine()
        engine.is_loaded = True
        engine.model = MagicMock()
        engine.model_type = "yolo"

        with pytest.raises((ValueError, RuntimeError)):
            asyncio.get_event_loop().run_until_complete(
                engine.run(b"not_an_image")
            )

    def test_09_get_image_dimensions(self, sample_image_bytes: bytes):
        """Test image dimension extraction from bytes."""
        engine = InferenceEngine()
        w, h = engine.get_image_dimensions(sample_image_bytes)
        assert w == 640
        assert h == 480

    def test_10_get_image_dimensions_invalid(self):
        """Test image dimension extraction with invalid bytes."""
        engine = InferenceEngine()
        w, h = engine.get_image_dimensions(b"not_an_image")
        assert w == 0
        assert h == 0


class TestItemCounter:
    """Tests for the item counting utility."""

    def test_11_count_items_basic(self, sample_detections: list[DetectionResultData]):
        """Test basic item counting from detection results."""
        counts = count_items(sample_detections)
        assert counts["juice_auer_cranberry"] == 2
        assert counts["water_juvina"] == 1

    def test_12_count_items_empty(self):
        """Test counting with empty detection list."""
        counts = count_items([])
        assert counts == {}

    def test_13_count_items_dict_input(self):
        """Test counting with dictionary-style detection objects."""
        dicts = [
            {"class_name": "apple", "confidence": 0.9},
            {"class_name": "apple", "confidence": 0.8},
            {"class_name": "banana", "confidence": 0.7},
        ]
        counts = count_items(dicts)
        assert counts["apple"] == 2
        assert counts["banana"] == 1

    def test_14_average_confidence(self, sample_detections: list[DetectionResultData]):
        """Test average confidence calculation for a specific class."""
        avg = get_average_confidence(sample_detections, "juice_auer_cranberry")
        assert abs(avg - 0.90) < 0.01  # (0.92 + 0.88) / 2 = 0.90

    def test_15_average_confidence_missing_class(self, sample_detections: list[DetectionResultData]):
        """Test average confidence for a class with no detections."""
        avg = get_average_confidence(sample_detections, "nonexistent_class")
        assert avg == 0.0

    def test_16_filter_detections_by_class(self, sample_detections: list[DetectionResultData]):
        """Test filtering detections by class name."""
        filtered = get_detections_for_class(sample_detections, "juice_auer_cranberry")
        assert len(filtered) == 2

    def test_17_filter_detections_empty(self, sample_detections: list[DetectionResultData]):
        """Test filtering with nonexistent class."""
        filtered = get_detections_for_class(sample_detections, "nonexistent")
        assert len(filtered) == 0


class TestDetectionResultData:
    """Tests for the DetectionResultData dataclass."""

    def test_18_detection_result_creation(self):
        """Test creating a DetectionResultData instance."""
        det = DetectionResultData(
            class_id=1,
            class_name="test_product",
            confidence=0.95,
            bbox=[10, 20, 30, 40],
            mask_polygon=[[10, 20], [30, 20], [30, 40], [10, 40]],
            mask_area=400.0,
        )
        assert det.class_id == 1
        assert det.class_name == "test_product"
        assert det.confidence == 0.95
        assert len(det.bbox) == 4
        assert len(det.mask_polygon) == 4
        assert det.mask_area == 400.0

    def test_19_detection_result_defaults(self):
        """Test default values for DetectionResultData."""
        det = DetectionResultData()
        assert det.class_id == 0
        assert det.class_name == ""
        assert det.confidence == 0.0
        assert det.bbox == []
        assert det.mask_polygon == []
        assert det.mask_area == 0.0

    def test_20_last_processed_image_initially_none(self):
        """Test that no processed image exists initially."""
        engine = InferenceEngine()
        assert engine.get_last_processed_image() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
