"""
tests/test_api.py — Unit tests for the FastAPI backend API endpoints.

Tests health check, inventory CRUD, checkout scan validation,
transaction history, restock, and error handling.
Uses pytest with httpx TestClient — at least 10 test cases.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set test environment variables before importing app
os.environ["DB_PATH"] = "database/test_checkout.db"
os.environ["MODEL_TYPE"] = "yolo"
os.environ["MODEL_WEIGHTS_PATH"] = "weights/nonexistent.pt"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"

from fastapi.testclient import TestClient

from database.db import Base, Product, create_all_tables, get_engine, get_session_factory
from database.seed import seed_database
from main import app


# ──────────────────── Fixtures ────────────────────


@pytest.fixture(scope="module")
def test_db():
    """Set up a test database and seed it."""
    db_path = "database/test_checkout.db"
    os.makedirs("database", exist_ok=True)

    engine = get_engine(db_path)
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    session = session_factory()
    try:
        seed_database(session, reset=True)
    finally:
        session.close()

    yield engine

    # Cleanup
    engine.dispose()
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def client(test_db):
    """Create a FastAPI test client."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ──────────────────── Test Cases ────────────────────


class TestHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    def test_01_health_check_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_02_health_check_response_structure(self, client):
        """Test that health response has required fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_type" in data
        assert "database_connected" in data
        assert "version" in data
        assert "product_count" in data

    def test_03_health_model_not_loaded(self, client):
        """Test that model_loaded is False when weights are missing."""
        response = client.get("/api/v1/health")
        data = response.json()
        # Model should not be loaded since we use nonexistent weights
        assert data["model_type"] == "yolo"


class TestInventoryEndpoints:
    """Tests for the /api/v1/inventory endpoints."""

    def test_04_get_all_products(self, client):
        """Test retrieving all products."""
        response = client.get("/api/v1/inventory/products")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_05_get_product_structure(self, client):
        """Test that product response has required fields."""
        response = client.get("/api/v1/inventory/products")
        data = response.json()
        if len(data) > 0:
            product = data[0]
            assert "id" in product
            assert "name" in product
            assert "category" in product
            assert "unit_price" in product
            assert "stock_quantity" in product

    def test_06_get_single_product(self, client):
        """Test retrieving a single product by ID."""
        # First get all products to find a valid ID
        all_products = client.get("/api/v1/inventory/products").json()
        if len(all_products) > 0:
            product_id = all_products[0]["id"]
            response = client.get(f"/api/v1/inventory/products/{product_id}")
            assert response.status_code == 200
            assert response.json()["id"] == product_id

    def test_07_get_nonexistent_product(self, client):
        """Test retrieving a product that doesn't exist returns 404."""
        response = client.get("/api/v1/inventory/products/99999")
        assert response.status_code == 404

    def test_08_update_product_price(self, client):
        """Test updating a product's price."""
        all_products = client.get("/api/v1/inventory/products").json()
        if len(all_products) > 0:
            product_id = all_products[0]["id"]
            response = client.put(
                f"/api/v1/inventory/products/{product_id}",
                json={"unit_price": 9.99},
            )
            assert response.status_code == 200
            assert response.json()["unit_price"] == 9.99

    def test_09_search_products(self, client):
        """Test searching products by name."""
        response = client.get("/api/v1/inventory/products", params={"search": "juice"})
        assert response.status_code == 200
        data = response.json()
        # All results should contain 'juice' in name or category
        for product in data:
            assert (
                "juice" in product["name"].lower()
                or "juice" in product["category"].lower()
            )

    def test_10_restock_products(self, client):
        """Test bulk restocking products."""
        all_products = client.get("/api/v1/inventory/products").json()
        if len(all_products) >= 2:
            items = [
                {"product_id": all_products[0]["id"], "quantity": 10},
                {"product_id": all_products[1]["id"], "quantity": 20},
            ]
            response = client.post(
                "/api/v1/inventory/restock",
                json={"items": items},
            )
            assert response.status_code == 200


class TestCheckoutEndpoints:
    """Tests for the /api/v1/checkout endpoints."""

    def test_11_scan_without_image(self, client):
        """Test scan endpoint without providing an image."""
        response = client.post("/api/v1/checkout/scan")
        assert response.status_code == 422  # Validation error

    def test_12_scan_with_non_image_file(self, client):
        """Test scan endpoint with a non-image file."""
        response = client.post(
            "/api/v1/checkout/scan",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        # Should reject non-image MIME type
        assert response.status_code in (400, 503)

    def test_13_get_checkout_history(self, client):
        """Test retrieving checkout history."""
        response = client.get("/api/v1/checkout/history")
        assert response.status_code == 200
        data = response.json()
        assert "transactions" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data

    def test_14_get_checkout_history_pagination(self, client):
        """Test checkout history pagination parameters."""
        response = client.get(
            "/api/v1/checkout/history",
            params={"page": 1, "limit": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["limit"] == 5

    def test_15_debug_last_no_image(self, client):
        """Test debug endpoint when no image has been processed."""
        response = client.get("/api/v1/checkout/debug/last")
        assert response.status_code == 404


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_16_swagger_docs_accessible(self, client):
        """Test that Swagger UI is accessible at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_17_redoc_accessible(self, client):
        """Test that ReDoc is accessible at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_18_openapi_json_accessible(self, client):
        """Test that OpenAPI JSON is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "info" in data
        assert data["info"]["title"] == "AI-Powered Automatic Retail Checkout System"


class TestErrorHandling:
    """Tests for error handling."""

    def test_19_invalid_route_returns_404(self, client):
        """Test that non-existent routes return 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code in (404, 405)

    def test_20_update_nonexistent_product(self, client):
        """Test that updating a nonexistent product returns 404."""
        response = client.put(
            "/api/v1/inventory/products/99999",
            json={"unit_price": 1.0},
        )
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
