"""
models/schemas.py — Pydantic request and response models for the AI Checkout API.

Defines the exact JSON contracts for all API endpoints including checkout scan
responses, product schemas, inventory operations, and health checks.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────── Detection / Inference ─────────────────────────


class DetectionResult(BaseModel):
    """A single instance segmentation detection from the model."""

    class_id: int = Field(..., description="Numeric class ID from the model")
    class_name: str = Field(..., description="Human-readable class/category name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: list[float] = Field(..., description="Bounding box as [x1, y1, x2, y2]")
    mask_polygon: list[list[float]] = Field(
        ..., description="Segmentation mask as list of [x, y] polygon points"
    )
    mask_area: float = Field(0.0, description="Area of the segmentation mask in pixels")


# ──────────────────────── Checkout Scan Response ────────────────────────


class CheckoutItem(BaseModel):
    """A single line item in the checkout receipt."""

    product_id: int = Field(..., description="Database product ID")
    name: str = Field(..., description="Display name of the product")
    quantity: int = Field(..., ge=1, description="Number of instances detected")
    unit_price: float = Field(..., ge=0.0, description="Price per unit in USD")
    subtotal: float = Field(..., ge=0.0, description="quantity × unit_price")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence across detections")
    segmentation: list[list[float]] = Field(
        default_factory=list,
        description="Polygon coordinates for mask overlay [[x1,y1, x2,y2, ...], ...]",
    )
    mask_color: str = Field("#FF5722", description="Hex colour for the mask overlay")


class ImageDimensions(BaseModel):
    """Dimensions of the processed image."""

    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class CheckoutResponse(BaseModel):
    """Complete response from the POST /api/v1/checkout/scan endpoint."""

    transaction_id: str = Field(..., description="UUID of the created transaction")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of the transaction")
    processing_time_ms: float = Field(..., description="Total inference + processing time in ms")
    items: list[CheckoutItem] = Field(default_factory=list, description="Detected and priced items")
    subtotal: float = Field(0.0, description="Sum of all item subtotals before tax")
    tax_rate: float = Field(0.10, description="Tax rate applied (default 10%)")
    tax_amount: float = Field(0.0, description="Calculated tax amount")
    total: float = Field(0.0, description="Grand total including tax")
    items_not_found: list[str] = Field(
        default_factory=list,
        description="Item names that were detected but not found in inventory or out of stock",
    )
    image_dimensions: ImageDimensions = Field(..., description="Dimensions of the processed image")


# ────────────────────────── Product Schemas ──────────────────────────


class ProductSchema(BaseModel):
    """Public-facing product representation."""

    id: int
    name: str
    category: str
    unit_price: float
    stock_quantity: int
    barcode: Optional[str] = None
    image_url: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ProductUpdateRequest(BaseModel):
    """Request body for updating a product's price or stock."""

    unit_price: Optional[float] = Field(None, ge=0.0, description="New price in USD")
    stock_quantity: Optional[int] = Field(None, ge=0, description="New stock quantity")


class RestockItem(BaseModel):
    """A single item in a bulk restock request."""

    product_id: int = Field(..., description="ID of the product to restock")
    quantity: int = Field(..., gt=0, description="Quantity to add to stock")


class RestockRequest(BaseModel):
    """Request body for the bulk restock endpoint."""

    items: list[RestockItem] = Field(..., min_length=1, description="List of products and quantities to restock")


class RestockResponse(BaseModel):
    """Response from the bulk restock endpoint."""

    message: str = "Restock completed successfully"
    updated_products: list[ProductSchema] = Field(default_factory=list)


# ──────────────────────── Transaction Schemas ────────────────────────


class TransactionItemSchema(BaseModel):
    """A line item within a transaction."""

    id: int
    product_id: int
    product_name: Optional[str] = None
    quantity: int
    unit_price_at_sale: float
    subtotal: float

    class Config:
        from_attributes = True


class TransactionSchema(BaseModel):
    """Full transaction record with line items."""

    id: str
    timestamp: datetime
    total_amount: float
    tax_amount: float
    subtotal: float
    processing_time_ms: Optional[float] = None
    items: list[TransactionItemSchema] = Field(default_factory=list)

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    """Paginated list of transactions."""

    transactions: list[TransactionSchema] = Field(default_factory=list)
    total: int = Field(0, description="Total number of transactions")
    page: int = Field(1, description="Current page number")
    limit: int = Field(20, description="Items per page")


# ────────────────────────── Health Check ─────────────────────────────


class HealthResponse(BaseModel):
    """Response from the GET /api/v1/health endpoint."""

    status: str = Field("ok", description="Overall system status")
    model_loaded: bool = Field(False, description="Whether the ML model is loaded in memory")
    model_type: str = Field("unknown", description="Active model type (yolo or maskrcnn)")
    database_connected: bool = Field(False, description="Whether the database is accessible")
    version: str = Field("1.0.0", description="API version")
    product_count: int = Field(0, description="Number of products in database")


# ────────────────────────── Auth Schemas ──────────────────────────────


class LoginRequest(BaseModel):
    """Request body for admin login."""

    username: str = Field(..., min_length=3, max_length=100, description="Admin username")
    password: str = Field(..., min_length=8, max_length=128, description="Admin password")


class LoginResponse(BaseModel):
    """Response from the login endpoint — returns session_token for MFA step."""

    message: str = Field("OTP sent to registered email", description="Status message")
    session_token: str = Field(..., description="Session token to use when submitting OTP")
    email_hint: str = Field(..., description="Masked email hint (e.g. m***@gmail.com)")
    otp_expiry_minutes: int = Field(5, description="Minutes until OTP expires")


class OTPVerifyRequest(BaseModel):
    """Request body for OTP verification (step 2 of MFA login)."""

    session_token: str = Field(..., description="Session token from login step")
    otp_code: str = Field(..., min_length=6, max_length=6, description="6-digit OTP code")


class OTPVerifyResponse(BaseModel):
    """Response from OTP verification — returns JWT access token."""

    message: str = Field("Login successful", description="Status message")
    access_token: str = Field(..., description="JWT access token for authenticated requests")
    token_type: str = Field("bearer", description="Token type")
    expires_in_minutes: int = Field(60, description="Token expiry in minutes")
    username: str = Field(..., description="Authenticated admin username")


class TokenValidationResponse(BaseModel):
    """Response from token validation endpoint."""

    valid: bool = Field(..., description="Whether the token is valid")
    username: str = Field("", description="Username from the token")
    user_id: int = Field(0, description="User ID from the token")


# ────────────────────────── Error Schemas ────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error type or category")
    detail: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., description="HTTP status code")
