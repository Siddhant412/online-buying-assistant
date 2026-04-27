from fastapi.testclient import TestClient

from app.api import app


def test_healthz_returns_status_and_request_id() -> None:
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"]


def test_v1_products_route_exists() -> None:
    paths = {route.path for route in app.routes}

    assert "/products" in paths
    assert "/api/v1/products" in paths
    assert "/api/v1/ask" in paths
