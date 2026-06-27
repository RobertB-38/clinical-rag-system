from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_rejects_short_question():
    # Validation (422) happens before the handler, so no pipeline is built.
    response = client.post("/v1/query", json={"question": "Hi"})
    assert response.status_code == 422
