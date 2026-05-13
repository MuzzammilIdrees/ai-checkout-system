# API Documentation — AI Checkout System

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

No authentication required (development mode).

---

## Endpoints

### 1. Health Check

**`GET /api/v1/health`**

Returns system health status including model loading state and database connectivity.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "yolo",
  "database_connected": true,
  "version": "1.0.0",
  "product_count": 60
}
```

---

### 2. Checkout Scan

**`POST /api/v1/checkout/scan`**

Main inference endpoint. Upload an image of groceries for instance segmentation, pricing, and checkout.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | Yes | JPEG or PNG image (max 20MB) |

**Response (200 OK):**
```json
{
  "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2025-01-01T12:00:00Z",
  "processing_time_ms": 87.4,
  "items": [
    {
      "product_id": 12,
      "name": "Coca-Cola 330ml",
      "quantity": 2,
      "unit_price": 1.50,
      "subtotal": 3.00,
      "confidence": 0.94,
      "segmentation": [[100, 200, 300, 200, 300, 400, 100, 400]],
      "mask_color": "#FF5722"
    }
  ],
  "subtotal": 3.00,
  "tax_rate": 0.10,
  "tax_amount": 0.30,
  "total": 3.30,
  "items_not_found": [],
  "image_dimensions": {"width": 1920, "height": 1080}
}
```

**Error Responses:**
- `400` — Invalid image format or too large
- `503` — Model not loaded

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/checkout/scan" \
  -F "image=@test_image.jpg"
```

---

### 3. Checkout History

**`GET /api/v1/checkout/history`**

Returns a paginated list of past checkout transactions.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number (1-indexed) |
| `limit` | int | 20 | Items per page (1-100) |

**Response (200 OK):**
```json
{
  "transactions": [
    {
      "id": "uuid-string",
      "timestamp": "2025-01-01T12:00:00Z",
      "total_amount": 15.40,
      "tax_amount": 1.40,
      "subtotal": 14.00,
      "processing_time_ms": 92.3,
      "items": [
        {
          "id": 1,
          "product_id": 5,
          "product_name": "NÖM Whole Milk 1L",
          "quantity": 2,
          "unit_price_at_sale": 1.89,
          "subtotal": 3.78
        }
      ]
    }
  ],
  "total": 42,
  "page": 1,
  "limit": 20
}
```

---

### 4. Debug — Last Processed Image

**`GET /api/v1/checkout/debug/last`**

Returns the last processed image with segmentation masks drawn on it as a JPEG.

**Response:** JPEG image (`image/jpeg`)

**Error:** `404` if no image has been processed yet.

---

### 5. Get All Products

**`GET /api/v1/inventory/products`**

Returns all 60 products with current stock levels and prices.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `search` | string | Search by name or category |
| `category` | string | Filter by category |
| `in_stock` | bool | Filter by stock availability |

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "name": "Auer Cranberry Juice 1L",
    "category": "Beverages",
    "unit_price": 3.49,
    "stock_quantity": 67,
    "barcode": "D2S-JUICE-AUER-CRANBER",
    "image_url": null,
    "created_at": "2025-01-01T00:00:00"
  }
]
```

---

### 6. Get Single Product

**`GET /api/v1/inventory/products/{product_id}`**

Returns a single product's details.

**Response (200 OK):** Same structure as single item in product list.

**Error:** `404` if product not found.

---

### 7. Update Product

**`PUT /api/v1/inventory/products/{product_id}`**

Update a product's price or stock quantity.

**Request Body:**
```json
{
  "unit_price": 4.99,
  "stock_quantity": 50
}
```

Both fields are optional — only send what you want to update.

**Response (200 OK):** Updated product object.

---

### 8. Bulk Restock

**`POST /api/v1/inventory/restock`**

Add stock quantities to multiple products at once.

**Request Body:**
```json
{
  "items": [
    {"product_id": 1, "quantity": 50},
    {"product_id": 2, "quantity": 30}
  ]
}
```

**Response (200 OK):**
```json
{
  "message": "Successfully restocked 2 products.",
  "updated_products": [...]
}
```

---

## Error Response Format

All errors follow this structure:

```json
{
  "error": "NotFound",
  "detail": "Product with id=99999 not found.",
  "status_code": 404
}
```

## Rate Limiting

- Default: **10 requests per minute** per IP
- Health endpoint: 50 requests per minute
- Configurable via `RATE_LIMIT_PER_MINUTE` in `.env`

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json
