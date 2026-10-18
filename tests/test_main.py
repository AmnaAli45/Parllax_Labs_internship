from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app, raise_server_exceptions=False)




def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}




def test_ask_with_valid_query_returns_200():
    response = client.post("/ask", json={"query": "What is deep learning?"})
    assert response.status_code == 200


def test_ask_response_has_expected_fields():
    response = client.post("/ask", json={"query": "What is deep learning?"})
    data = response.json()
    assert "answer" in data
    assert "status" in data
    assert "retrieval_ms" in data
    assert "generation_ms" in data
    assert "total_ms" in data


def test_ask_uses_default_top_k_when_not_given():
    response = client.post("/ask", json={"query": "What is deep learning?"})
    assert response.status_code == 200



def test_ask_with_empty_query_returns_400():
    response = client.post("/ask", json={"query": "   "})
    assert response.status_code == 400


def test_ask_with_top_k_too_high_returns_400():
    response = client.post("/ask", json={"query": "hello", "top_k": 100})
    assert response.status_code == 400


def test_ask_with_top_k_zero_returns_400():
    response = client.post("/ask", json={"query": "hello", "top_k": 0})
    assert response.status_code == 400




def test_ask_without_query_field_returns_422():
    response = client.post("/ask", json={"top_k": 3})
    assert response.status_code == 422


def test_ask_with_wrong_type_for_top_k_returns_422():
    response = client.post("/ask", json={"query": "hello", "top_k": "not_a_number"})
    assert response.status_code == 422


def test_ask_with_no_body_returns_422():
    response = client.post("/ask")
    assert response.status_code == 422




def test_ask_with_crash_trigger_returns_500():
    response = client.post("/ask", json={"query": "trigger crash"})
    assert response.status_code == 500


def test_ask_500_response_does_not_leak_internal_details():
    response = client.post("/ask", json={"query": "trigger crash"})
    data = response.json()
    assert "RuntimeError" not in str(data)
    assert "error" in data




def test_response_has_request_id_header():
    response = client.get("/health")
    assert "X-Request-ID" in response.headers