# Architecture — AI-Powered Automatic Retail Checkout System

## System Overview

The system consists of four main components connected through HTTP APIs:

```
┌─────────────────┐      HTTP/JSON      ┌──────────────────────┐
│                 │  ◄──────────────►   │                      │
│  Flutter POS    │                     │   FastAPI Backend     │
│  Application    │   multipart/form    │                      │
│                 │  ──────────────►    │   ┌───────────────┐  │
│  • Checkout     │                     │   │ Inference     │  │
│  • Receipt      │   JSON response     │   │ Engine        │  │
│  • Inventory    │  ◄──────────────    │   │ (Singleton)   │  │
│  • History      │                     │   └───────┬───────┘  │
│                 │                     │           │           │
└─────────────────┘                     │   ┌───────▼───────┐  │
                                        │   │ SQLite DB     │  │
                                        │   │ (SQLAlchemy)  │  │
                                        │   └───────────────┘  │
                                        └──────────────────────┘

┌─────────────────────────────────────┐
│  Training Pipeline (Offline)        │
│                                     │
│  MVTec D2S ──► data_preparation.py  │
│             ──► train.py            │
│             ──► validate.py         │
│             ──► best.pt weights     │
└─────────────────────────────────────┘
```

## Component Architecture

### 1. Deep Learning Model

**Primary: YOLOv8-seg (Ultralytics)**
- Architecture: YOLOv8n-seg (nano) for speed, YOLOv8m-seg for accuracy
- Input: 640×640 RGB images
- Output: Per-instance class labels, confidence scores, bounding boxes, and polygon segmentation masks
- Transfer learning from COCO-pretrained weights

**Alternative: Mask R-CNN (Detectron2)**
- Architecture: ResNet-50-FPN backbone
- Input: Variable size (shorter side 800px)
- Switchable via `MODEL_TYPE` environment variable

**Why Instance Segmentation over Object Detection?**
- Pixel-accurate masks handle overlapping items on a checkout counter
- Masks provide precise visual feedback in the POS overlay
- Segmentation polygons are returned to the frontend for rendering

### 2. FastAPI Backend

**Design Decisions:**
- **Singleton InferenceEngine**: Model loaded once at startup via FastAPI lifespan event — zero reload overhead per request
- **Asyncio Lock**: Thread-safe inference for concurrent requests without race conditions
- **SQLAlchemy ORM**: Type-safe database operations with atomic transaction support
- **Pydantic v2**: Fast JSON serialization/deserialization with strict type validation
- **SlowAPI Rate Limiting**: Prevents abuse without external dependencies

**Request Flow:**
1. Flutter sends multipart image → FastAPI validates (size, MIME type)
2. Image decoded, resized if needed → InferenceEngine.run()
3. YOLOv8-seg produces detections → item_counter aggregates by class
4. Price lookup in SQLite → inventory deduction (atomic)
5. Transaction persisted → JSON response with masks + receipt

### 3. SQLite Database

**Tables:**
- `products` — 60 D2S categories with names, prices, stock
- `transactions` — Checkout records with UUID, totals, timestamps
- `transaction_items` — Line items per transaction
- `inventory_log` — Audit trail for all stock changes (SALE/RESTOCK)

**Business Logic:**
- Inventory deduction is atomic (all-or-nothing per transaction)
- Stock can never go below 0 (clamped with warning log)
- Out-of-stock items reported in `items_not_found`

### 4. Flutter Frontend

**State Management: Riverpod 2.x**
- `checkoutProvider` — Manages scan state (loading, response, error, image)
- `inventoryProvider` — FutureProvider with auto-dispose caching
- `transactionHistoryProvider` — Paginated state with infinite scroll

**Key Widgets:**
- `SegmentationOverlay` — CustomPainter with RepaintBoundary for 60fps mask rendering
- `ReceiptItemTile` — Staggered slide-in animations
- `CameraPreviewWidget` — InteractiveViewer with pinch-to-zoom

**Design System:**
- Primary: #1565C0 (Deep Blue) — POS terminal feel
- Accent: #FF6F00 (Amber) — Call-to-action buttons
- Typography: Inter (UI), Source Code Pro (receipt amounts)
- 20-colour palette for deterministic mask colouring by class_id

## Data Flow Diagram

```
Step 1: Camera Capture
  Flutter app captures JPEG frame (camera or image picker)
  Image compressed to max 2MB

Step 2: HTTP POST /api/v1/checkout/scan
  Flutter ApiService sends multipart/form-data to FastAPI
  Dio handles retries (max 3, exponential backoff)

Step 3: Image Preprocessing
  FastAPI validates image (size, MIME type)
  Resizes if longer side > 4096px

Step 4: Model Inference
  InferenceEngine.run() → YOLOv8-seg or Mask R-CNN
  Returns List[DetectionResult]

Step 5: Item Counting
  item_counter.count_items() → {class_name: quantity}

Step 6: Price Lookup
  Query SQLite products table by class name
  Build line items with unit_price and subtotal

Step 7: Inventory Deduction
  Atomically decrement stock_quantity
  Write audit entries to inventory_log

Step 8: Transaction Persistence
  Write transaction + transaction_items rows
  Generate UUID, timestamp, totals

Step 9: JSON Response
  Serialize CheckoutResponse → JSON → 200 OK

Step 10: Flutter Rendering
  checkoutProvider updates state
  SegmentationOverlay draws masks
  Receipt panel animates items in
```

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| DL Framework | PyTorch + Ultralytics | Best-in-class YOLOv8 ecosystem |
| API | FastAPI | Async support, auto-docs, Pydantic validation |
| Database | SQLite + SQLAlchemy | Lightweight, zero config, ORM for type safety |
| Frontend | Flutter + Riverpod | Cross-platform, reactive state, 60fps rendering |
| HTTP Client | Dio | Interceptors, retry logic, multipart uploads |

## Security Considerations

- CORS configurable via `.env` (currently `*` for development)
- Rate limiting (10 req/min per IP) prevents abuse
- Image validation: size limit (20MB), MIME type check
- No credentials stored — all config via environment variables
- SQL injection prevented by SQLAlchemy ORM parameterized queries
