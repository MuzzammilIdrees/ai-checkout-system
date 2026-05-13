"""
routers/inventory.py — Inventory management endpoints for the AI Checkout System.

Provides CRUD operations for the 60 D2S products including stock levels,
pricing updates, and bulk restock functionality.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session

from database.db import InventoryLog, Product
from models.schemas import (
    ErrorResponse,
    ProductSchema,
    ProductUpdateRequest,
    RestockItem,
    RestockRequest,
    RestockResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/inventory", tags=["Inventory"])


@router.get(
    "/products",
    response_model=list[ProductSchema],
    summary="Get all products",
    description="Returns all 60 products with current stock levels and prices.",
)
async def get_all_products(
    search: Optional[str] = Query(None, description="Search products by name or category"),
    category: Optional[str] = Query(None, description="Filter by category"),
    in_stock: Optional[bool] = Query(None, description="Filter by stock availability"),
) -> list[ProductSchema]:
    """
    Retrieve all products from the inventory, with optional filtering.

    Args:
        search: Optional search term to filter by name or category (case-insensitive).
        category: Optional category filter.
        in_stock: If True, only return products with stock > 0.

    Returns:
        List of ProductSchema objects.
    """
    from main import get_db_session

    db = get_db_session()
    try:
        query = db.query(Product)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (Product.name.ilike(search_term)) | (Product.category.ilike(search_term))
            )

        if category:
            query = query.filter(Product.category.ilike(f"%{category}%"))

        if in_stock is True:
            query = query.filter(Product.stock_quantity > 0)
        elif in_stock is False:
            query = query.filter(Product.stock_quantity == 0)

        products = query.order_by(Product.category, Product.name).all()

        return [
            ProductSchema(
                id=p.id,
                name=p.name,
                category=p.category,
                unit_price=p.unit_price,
                stock_quantity=p.stock_quantity,
                barcode=p.barcode,
                image_url=p.image_url,
                created_at=p.created_at,
            )
            for p in products
        ]
    finally:
        db.close()


@router.get(
    "/products/{product_id}",
    response_model=ProductSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Get a single product",
    description="Returns a single product's details by ID.",
)
async def get_product(product_id: int) -> ProductSchema:
    """
    Retrieve a single product by its database ID.

    Args:
        product_id: The product's primary key.

    Returns:
        ProductSchema for the requested product.

    Raises:
        HTTPException 404: Product not found.
    """
    from main import get_db_session

    db = get_db_session()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product with id={product_id} not found.")

        return ProductSchema(
            id=product.id,
            name=product.name,
            category=product.category,
            unit_price=product.unit_price,
            stock_quantity=product.stock_quantity,
            barcode=product.barcode,
            image_url=product.image_url,
            created_at=product.created_at,
        )
    finally:
        db.close()


@router.put(
    "/products/{product_id}",
    response_model=ProductSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Update a product",
    description="Update a product's price or stock quantity.",
)
async def update_product(product_id: int, update: ProductUpdateRequest) -> ProductSchema:
    """
    Update a product's price and/or stock quantity.

    Args:
        product_id: The product's primary key.
        update: ProductUpdateRequest with optional unit_price and stock_quantity.

    Returns:
        Updated ProductSchema.

    Raises:
        HTTPException 404: Product not found.
    """
    from main import get_db_session

    db = get_db_session()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product with id={product_id} not found.")

        if update.unit_price is not None:
            old_price = product.unit_price
            product.unit_price = update.unit_price
            logger.info(f"💰 Price updated for '{product.name}': ${old_price:.2f} → ${update.unit_price:.2f}")

        if update.stock_quantity is not None:
            old_stock = product.stock_quantity
            product.stock_quantity = update.stock_quantity
            logger.info(f"📦 Stock updated for '{product.name}': {old_stock} → {update.stock_quantity}")

            # Log the inventory change
            delta = update.stock_quantity - old_stock
            inv_log = InventoryLog(
                product_id=product.id,
                change_type="RESTOCK" if delta > 0 else "SALE",
                quantity_delta=delta,
                timestamp=datetime.now(timezone.utc),
            )
            db.add(inv_log)

        db.commit()
        db.refresh(product)

        return ProductSchema(
            id=product.id,
            name=product.name,
            category=product.category,
            unit_price=product.unit_price,
            stock_quantity=product.stock_quantity,
            barcode=product.barcode,
            image_url=product.image_url,
            created_at=product.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post(
    "/restock",
    response_model=RestockResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Bulk restock products",
    description="Add stock quantities to multiple products at once.",
)
async def restock_products(request: RestockRequest) -> RestockResponse:
    """
    Bulk restock: add quantities to multiple products in a single transaction.

    All restocks are applied atomically — if any item fails, all changes
    are rolled back.

    Args:
        request: RestockRequest containing a list of product_id + quantity pairs.

    Returns:
        RestockResponse with updated product details.

    Raises:
        HTTPException 400: Invalid product ID in the request.
    """
    from main import get_db_session

    db = get_db_session()
    try:
        updated_products: list[ProductSchema] = []
        now = datetime.now(timezone.utc)

        for restock_item in request.items:
            product = db.query(Product).filter(Product.id == restock_item.product_id).first()
            if not product:
                raise HTTPException(
                    status_code=400,
                    detail=f"Product with id={restock_item.product_id} not found.",
                )

            old_stock = product.stock_quantity
            product.stock_quantity += restock_item.quantity

            # Create inventory log entry
            inv_log = InventoryLog(
                product_id=product.id,
                change_type="RESTOCK",
                quantity_delta=restock_item.quantity,
                timestamp=now,
            )
            db.add(inv_log)

            logger.info(
                f"📦 Restocked '{product.name}': {old_stock} → {product.stock_quantity} "
                f"(+{restock_item.quantity})"
            )

            updated_products.append(
                ProductSchema(
                    id=product.id,
                    name=product.name,
                    category=product.category,
                    unit_price=product.unit_price,
                    stock_quantity=product.stock_quantity,
                    barcode=product.barcode,
                    image_url=product.image_url,
                    created_at=product.created_at,
                )
            )

        db.commit()

        return RestockResponse(
            message=f"Successfully restocked {len(updated_products)} products.",
            updated_products=updated_products,
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Restock failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
