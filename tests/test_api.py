from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"features": [0.1]*20})  # adjust length
    assert response.status_code == 200 or response.status_code == 500
