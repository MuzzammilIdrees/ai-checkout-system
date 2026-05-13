"""
utils/item_counter.py — Shared quantity-counting utility for the AI Checkout System.

Groups detected instances by their class label and returns a dictionary mapping
class_name → quantity. Used by both the training prediction scripts and the
backend API checkout flow.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def count_items(detections: list[Any]) -> dict[str, int]:
    """
    Aggregate detection results by class label to count item quantities.

    Groups all detected instances by their class_name attribute and counts
    how many instances of each class were found.

    Args:
        detections: List of detection objects. Each object must have a
                    'class_name' attribute (or key if dict).

    Returns:
        Dictionary mapping class_name → quantity (count of instances).

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Det:
        ...     class_name: str
        ...     confidence: float
        >>> dets = [Det("apple", 0.9), Det("banana", 0.8), Det("apple", 0.95)]
        >>> count_items(dets)
        {'apple': 2, 'banana': 1}
    """
    item_counts: dict[str, int] = {}

    for detection in detections:
        # Support both object attribute and dict key access
        if isinstance(detection, dict):
            class_name = detection.get("class_name", "unknown")
        elif hasattr(detection, "class_name"):
            class_name = detection.class_name
        else:
            logger.warning(f"⚠️ Detection object has no 'class_name': {detection}")
            continue

        if not class_name or class_name.strip() == "":
            logger.warning("⚠️ Empty class_name encountered, skipping.")
            continue

        item_counts[class_name] = item_counts.get(class_name, 0) + 1

    logger.info(f"📊 Item counts: {item_counts}")
    return item_counts


def get_average_confidence(detections: list[Any], class_name: str) -> float:
    """
    Calculate the average confidence score for a given class across all detections.

    Args:
        detections: List of detection objects with class_name and confidence attributes.
        class_name: The class name to calculate average confidence for.

    Returns:
        Average confidence as a float between 0.0 and 1.0. Returns 0.0 if no
        matching detections are found.
    """
    confidences: list[float] = []

    for detection in detections:
        if isinstance(detection, dict):
            name = detection.get("class_name", "")
            conf = detection.get("confidence", 0.0)
        elif hasattr(detection, "class_name") and hasattr(detection, "confidence"):
            name = detection.class_name
            conf = detection.confidence
        else:
            continue

        if name == class_name:
            confidences.append(float(conf))

    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)


def get_detections_for_class(detections: list[Any], class_name: str) -> list[Any]:
    """
    Filter detections to return only those matching a specific class name.

    Args:
        detections: List of detection objects.
        class_name: The class name to filter by.

    Returns:
        List of detections matching the specified class name.
    """
    result = []
    for detection in detections:
        if isinstance(detection, dict):
            name = detection.get("class_name", "")
        elif hasattr(detection, "class_name"):
            name = detection.class_name
        else:
            continue

        if name == class_name:
            result.append(detection)

    return result
