"""
routers/checkout.py — Checkout endpoints for the AI Checkout System.

Handles the main scan workflow: image upload → inference → price lookup →
inventory deduction → transaction persistence → JSON response.
Also provides transaction history and a debug endpoint.
"""

import io
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database.db import InventoryLog, Product, Transaction, TransactionItem
from models.inference import InferenceEngine
from models.schemas import (
    CheckoutItem,
    CheckoutResponse,
    ErrorResponse,
    ImageDimensions,
    TransactionItemSchema,
    TransactionListResponse,
    TransactionSchema,
)
from utils.item_counter import count_items, get_average_confidence, get_detections_for_class

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/checkout", tags=["Checkout"])

# ──────────────── 20-colour palette for mask overlays ────────────────

MASK_COLORS = [
    "#FF5722", "#2196F3", "#4CAF50", "#FFC107", "#9C27B0",
    "#00BCD4", "#FF5733", "#3F51B5", "#8BC34A", "#FF9800",
    "#E91E63", "#009688", "#795548", "#FFEB3B", "#673AB7",
    "#F44336", "#03A9F4", "#CDDC39", "#607D8B", "#FF7043",
]


def get_db(request) -> Session:
    """Extract database session from request state (set by main.py dependency)."""
    return request.state.db


@router.post(
    "/scan",
    response_model=CheckoutResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Scan groceries for checkout",
    description=(
        "Upload an image of groceries on the checkout counter. The model performs "
        "instance segmentation, counts items, looks up prices, deducts inventory, "
        "and returns a complete checkout receipt."
    ),
)
async def scan_checkout(
    image: UploadFile = File(..., description="JPEG or PNG image of groceries"),
    db: Session = Depends(lambda: None),  # Overridden at runtime
) -> CheckoutResponse:
    """
    Main checkout scan endpoint.

    Accepts a multipart/form-data image, runs instance segmentation inference,
    looks up prices in the inventory database, atomically deducts stock, persists
    the transaction, and returns a complete receipt with mask polygons.

    Args:
        image: Uploaded image file (JPEG or PNG).
        db: SQLAlchemy database session.

    Returns:
        CheckoutResponse with itemised receipt, segmentation data, and totals.

    Raises:
        HTTPException 400: Invalid image format or too large.
        HTTPException 503: Model not loaded.
    """
    start_time = time.perf_counter()

    # === Validate image ===
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Only JPEG and PNG are accepted.",
        )

    image_bytes = await image.read()
    max_size_mb = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))
    if len(image_bytes) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({len(image_bytes) / 1024 / 1024:.1f}MB). Maximum is {max_size_mb}MB.",
        )

    # === Run inference ===
    engine = InferenceEngine()
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model and place weights in the weights/ directory.",
        )

    try:
        detections = await engine.run(image_bytes)
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # === Get image dimensions ===
    img_w, img_h = engine.get_image_dimensions(image_bytes)

    # === Count items ===
    item_counts = count_items(detections)

    # === Look up prices and build receipt ===
    # Get database session from the app state
    from main import get_db_session
    db = get_db_session()

    checkout_items: list[CheckoutItem] = []
    items_not_found: list[str] = []
    total_subtotal = 0.0

    try:
        for class_name, quantity in item_counts.items():
            # Look up product by matching the class_name to product names
            product = _find_product_by_class_name(db, class_name)

            if product is None:
                logger.warning(f"⚠️ Product not found in database: '{class_name}'")
                items_not_found.append(class_name)
                continue

            if product.stock_quantity <= 0:
                logger.warning(f"⚠️ Product out of stock: '{product.name}' (stock=0)")
                items_not_found.append(f"{product.name} (out of stock)")
                continue

            # Clamp quantity to available stock
            actual_qty = min(quantity, product.stock_quantity)
            if actual_qty < quantity:
                logger.warning(
                    f"⚠️ Insufficient stock for '{product.name}': "
                    f"detected={quantity}, available={product.stock_quantity}, using={actual_qty}"
                )

            line_subtotal = round(actual_qty * product.unit_price, 2)
            avg_confidence = get_average_confidence(detections, class_name)

            # Collect all mask polygons for this class
            class_detections = get_detections_for_class(detections, class_name)
            all_polygons = []
            for det in class_detections:
                if hasattr(det, "mask_polygon") and det.mask_polygon:
                    flat_polygon = []
                    for point in det.mask_polygon:
                        flat_polygon.extend(point)
                    all_polygons.append(flat_polygon)

            mask_color = MASK_COLORS[product.id % len(MASK_COLORS)]

            checkout_items.append(
                CheckoutItem(
                    product_id=product.id,
                    name=product.name,
                    quantity=actual_qty,
                    unit_price=product.unit_price,
                    subtotal=line_subtotal,
                    confidence=round(avg_confidence, 2),
                    segmentation=all_polygons,
                    mask_color=mask_color,
                )
            )
            total_subtotal += line_subtotal

        # === Calculate totals ===
        tax_rate = 0.10
        tax_amount = round(total_subtotal * tax_rate, 2)
        total = round(total_subtotal + tax_amount, 2)

        # === Create transaction ===
        transaction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 1)

        transaction = Transaction(
            id=transaction_id,
            timestamp=now,
            total_amount=total,
            tax_amount=tax_amount,
            subtotal=total_subtotal,
            processing_time_ms=processing_time_ms,
        )
        db.add(transaction)

        # === Deduct inventory and create line items ===
        for item in checkout_items:
            # Create transaction item
            txn_item = TransactionItem(
                transaction_id=transaction_id,
                product_id=item.product_id,
                quantity=item.quantity,
                unit_price_at_sale=item.unit_price,
                subtotal=item.subtotal,
            )
            db.add(txn_item)

            # Deduct stock atomically
            product = db.query(Product).filter(Product.id == item.product_id).with_for_update().first()
            if product:
                product.stock_quantity = max(0, product.stock_quantity - item.quantity)

                # Create inventory log entry
                inv_log = InventoryLog(
                    product_id=item.product_id,
                    change_type="SALE",
                    quantity_delta=-item.quantity,
                    timestamp=now,
                    transaction_id=transaction_id,
                )
                db.add(inv_log)

        # Commit all changes atomically
        db.commit()

        logger.info(
            f"✅ Checkout complete: txn={transaction_id}, items={len(checkout_items)}, "
            f"total=${total:.2f}, time={processing_time_ms:.1f}ms"
        )

        return CheckoutResponse(
            transaction_id=transaction_id,
            timestamp=now,
            processing_time_ms=processing_time_ms,
            items=checkout_items,
            subtotal=total_subtotal,
            tax_rate=tax_rate,
            tax_amount=tax_amount,
            total=total,
            items_not_found=items_not_found,
            image_dimensions=ImageDimensions(width=img_w, height=img_h),
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Checkout transaction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transaction error: {str(e)}")
    finally:
        db.close()


def _find_product_by_class_name(db: Session, class_name: str) -> Optional[Product]:
    """
    Find a product in the database matching the given D2S class name.

    Tries multiple matching strategies:
    1. Exact name match
    2. Case-insensitive partial match
    3. Match against the D2S name mapping from seed data

    Args:
        db: SQLAlchemy session.
        class_name: Class name from the model prediction.

    Returns:
        Matching Product or None if not found.
    """
    # Strategy 1: Direct name match
    product = db.query(Product).filter(Product.name == class_name).first()
    if product:
        return product

    # Strategy 2: Case-insensitive partial match on name
    product = (
        db.query(Product)
        .filter(Product.name.ilike(f"%{class_name}%"))
        .first()
    )
    if product:
        return product

    # Strategy 3: Match via D2S seed mapping
    from database.seed import get_d2s_name_to_product_name_map

    d2s_map = get_d2s_name_to_product_name_map()
    product_name = d2s_map.get(class_name)
    if product_name:
        product = db.query(Product).filter(Product.name == product_name).first()
        if product:
            return product

    # Strategy 4: Match barcode field which contains d2s_name fragments
    sanitized = class_name.upper().replace("_", "-")[:20]
    product = (
        db.query(Product)
        .filter(Product.barcode.ilike(f"%{sanitized}%"))
        .first()
    )
    return product


@router.get(
    "/history",
    response_model=TransactionListResponse,
    summary="Get checkout transaction history",
    description="Returns a paginated list of past checkout transactions.",
)
async def get_checkout_history(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> TransactionListResponse:
    """
    Retrieve paginated checkout transaction history.

    Args:
        page: Page number (1-indexed).
        limit: Number of transactions per page.

    Returns:
        TransactionListResponse with paginated transaction data.
    """
    from main import get_db_session
    db = get_db_session()

    try:
        total = db.query(Transaction).count()
        offset = (page - 1) * limit

        transactions = (
            db.query(Transaction)
            .order_by(Transaction.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        transaction_schemas = []
        for txn in transactions:
            items = db.query(TransactionItem).filter(TransactionItem.transaction_id == txn.id).all()
            item_schemas = []
            for item in items:
                product = db.query(Product).filter(Product.id == item.product_id).first()
                item_schemas.append(
                    TransactionItemSchema(
                        id=item.id,
                        product_id=item.product_id,
                        product_name=product.name if product else "Unknown",
                        quantity=item.quantity,
                        unit_price_at_sale=item.unit_price_at_sale,
                        subtotal=item.subtotal,
                    )
                )

            transaction_schemas.append(
                TransactionSchema(
                    id=txn.id,
                    timestamp=txn.timestamp,
                    total_amount=txn.total_amount,
                    tax_amount=txn.tax_amount,
                    subtotal=txn.subtotal,
                    processing_time_ms=txn.processing_time_ms,
                    items=item_schemas,
                )
            )

        return TransactionListResponse(
            transactions=transaction_schemas,
            total=total,
            page=page,
            limit=limit,
        )
    finally:
        db.close()


@router.get(
    "/debug/last",
    summary="Get last processed image with masks",
    description=(
        "Returns the last image processed by the scan endpoint with segmentation "
        "masks drawn on it. Useful for demo and debugging purposes."
    ),
)
async def get_debug_last_image():
    """
    Serve the last processed image with masks drawn for demo/debugging.

    Returns:
        JPEG image as a streaming response, or 404 if no image has been processed.
    """
    engine = InferenceEngine()
    image = engine.get_last_processed_image()

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="No image has been processed yet. Send a scan request first.",
        )

    # Encode as JPEG
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode debug image.")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=debug_last.jpg"},
    )


# Required import at module level for os.getenv
import os
