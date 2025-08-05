"""
Test suite for ML Model Serving API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

def test_models_list():
    """Test listing available models"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_prediction():
    """Test making a prediction"""
    prediction_data = {
        "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "use_ab_testing": False
    }
    
    response = client.post("/predict", json=prediction_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "model_version" in data
    assert "processing_time" in data
    assert "request_id" in data

def test_prediction_with_model_version():
    """Test making a prediction with specific model version"""
    prediction_data = {
        "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "model_version": "v1.0",
        "use_ab_testing": False
    }
    
    response = client.post("/predict", json=prediction_data)
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.0"

def test_ab_test_status():
    """Test A/B testing status endpoint"""
    response = client.get("/ab-test/status")
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data
    assert "config" in data

def test_metrics():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data

def test_dashboard():
    """Test dashboard endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_invalid_prediction():
    """Test prediction with invalid data"""
    prediction_data = {
        "features": []  # Empty features should cause an error
    }
    
    response = client.post("/predict", json=prediction_data)
    assert response.status_code == 500

if __name__ == "__main__":
    pytest.main([__file__])
