import pytest
from fastapi.testclient import TestClient
from src.ragapp.api import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_query_200_and_shape(client):
    response = client.post("/query",json={"question":"What is the expense ratio of the funds?"})
    assert response.status_code == 200
    body = response.json()
    
    assert "question" in body
    assert "answer" in body
    assert isinstance(body["answer"],str)
    assert len(body["answer"]) > 0
    

def test_out_of_scope_trigger(client):
    response = client.post("/query", json={"question":"what is the address of the CEO?"})
    answer = response.json()["answer"].lower()
    
    assert response.status_code ==200
    assert "don't have enough information" in answer

def test_malformed_request_rejected(client):
    response = client.post("/query",json={"no_field":"testing the wrong args passed"})
    assert response.status_code == 422
    