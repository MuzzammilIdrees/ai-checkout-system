"""
main.py — FastAPI application entrypoint for the AI Checkout System.

Configures the app with:
- Lifespan event: loads ML model, creates DB tables, seeds data
- CORS middleware for cross-origin Flutter requests
- Request logging middleware with latency tracking
- Global exception handler returning structured JSON errors
- Rate limiting via SlowAPI
- Health check endpoint
- Mounted routers for checkout and inventory
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from database.db import create_all_tables, get_engine, get_session_factory
from database.seed import seed_database
from models.inference import InferenceEngine
from models.schemas import ErrorResponse, HealthResponse
from routers import auth, checkout, inventory

# Load environment variables
load_dotenv()

# ──────────────────────── Logging Setup ─────────────────────────────

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────── Database Setup ────────────────────────────

DB_PATH = os.getenv("DB_PATH", "database/checkout.db")

# Ensure database directory exists
db_dir = os.path.dirname(DB_PATH)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

_engine = get_engine(DB_PATH)
_session_factory = get_session_factory(_engine)


def get_db_session():
    """
    Create and return a new database session.

    Returns:
        A new SQLAlchemy Session instance. Caller is responsible for closing it.
    """
    return _session_factory()


# ──────────────────────── Rate Limiter ──────────────────────────────

rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
limiter = Limiter(key_func=get_remote_address)


# ──────────────────────── Lifespan Event ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.

    On startup:
    1. Creates all database tables if they don't exist.
    2. Seeds the database with 60 D2S products (idempotent).
    3. Loads the ML model into memory.

    On shutdown:
    - Disposes the database engine.
    """
    logger.info("🚀 Starting AI Checkout System...")

    # 1. Create database tables
    create_all_tables(_engine)
    logger.info("✅ Database tables created/verified.")

    # 2. Seed database (idempotent)
    session = _session_factory()
    try:
        seed_database(session, reset=False)
    except Exception as e:
        logger.error(f"❌ Database seeding failed: {e}")
    finally:
        session.close()

    # 3. Load ML model
    engine = InferenceEngine()
    engine.load_model()
    if engine.is_loaded:
        logger.info(f"✅ ML model loaded: type={engine.model_type}")
    else:
        logger.warning(
            "⚠️ ML model NOT loaded. The /checkout/scan endpoint will return 503. "
            "Place trained weights at the path specified in MODEL_WEIGHTS_PATH."
        )

    yield

    # Shutdown
    logger.info("🛑 Shutting down AI Checkout System...")
    _engine.dispose()


# ──────────────────────── FastAPI App ───────────────────────────────

app = FastAPI(
    title="AI-Powered Automatic Retail Checkout System",
    description=(
        "Instance segmentation-based grocery checkout API. "
        "Accepts images of groceries, identifies items using YOLOv8-seg / Mask R-CNN, "
        "and returns itemised receipts with segmentation masks."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ──────────────────────── Middleware ────────────────────────────────

# CORS
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_raw == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Log every HTTP request with method, path, status code, and latency.

    Args:
        request: Incoming HTTP request.
        call_next: Next middleware/handler in the chain.

    Returns:
        The HTTP response.
    """
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as e:
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.error(f"❌ {request.method} {request.url.path} — ERROR in {elapsed:.1f}ms: {e}")
        raise

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"{'✅' if response.status_code < 400 else '⚠️'} "
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)"
    )

    return response


# ──────────────────── Global Exception Handler ──────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler returning structured JSON error responses.

    Args:
        request: The incoming request that caused the error.
        exc: The unhandled exception.

    Returns:
        JSONResponse with error details and 500 status code.
    """
    logger.error(f"❌ Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail=str(exc),
            status_code=500,
        ).model_dump(),
    )


# ──────────────────────── Mount Routers ─────────────────────────────

app.include_router(auth.router)
app.include_router(checkout.router)
app.include_router(inventory.router)


# ──────────────────────── Health Endpoint ───────────────────────────

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns system health including model status, database connectivity, and version.",
)
@limiter.limit(f"{rate_limit_per_minute * 5}/minute")
async def health_check(request: Request) -> HealthResponse:
    """
    System health check endpoint.

    Returns model loaded status, database connection status, product count,
    and API version.

    Args:
        request: The incoming HTTP request.

    Returns:
        HealthResponse with system status information.
    """
    engine = InferenceEngine()

    # Check database
    db_connected = False
    product_count = 0
    try:
        session = _session_factory()
        from database.db import Product

        product_count = session.query(Product).count()
        db_connected = True
        session.close()
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")

    return HealthResponse(
        status="ok" if engine.is_loaded and db_connected else "degraded",
        model_loaded=engine.is_loaded,
        model_type=engine.model_type,
        database_connected=db_connected,
        version="1.0.0",
        product_count=product_count,
    )


# ──────────────────────── Run Directly ──────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # reload=True is only safe for local development, not production/Railway
    is_dev = os.getenv("RAILWAY_ENVIRONMENT") is None
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=is_dev,
        log_level=log_level.lower(),
    )
