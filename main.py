"""
Machine Learning Model Serving API
FastAPI-based ML service with model versioning and A/B testing
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import redis.asyncio as redis

from models.model_manager import ModelManager
from models.prediction_service import PredictionService
from monitoring.metrics_collector import MetricsCollector
from utils.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
model_manager = None
prediction_service = None
metrics_collector = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_manager, prediction_service, metrics_collector, redis_client
    
    # Startup
    logger.info("Starting ML API service...")
    
    # Initialize Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Initialize components
    settings = Settings()
    model_manager = ModelManager(settings)
    prediction_service = PredictionService(model_manager, redis_client)
    metrics_collector = MetricsCollector()
    
    # Load initial models
    await model_manager.load_models()
    
    logger.info("ML API service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML API service...")
    if redis_client:
        await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Machine Learning API with model versioning and A/B testing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]
    model_version: Optional[str] = None
    use_ab_testing: bool = True

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    confidence: Optional[float] = None
    processing_time: float
    request_id: str

class ModelInfo(BaseModel):
    version: str
    name: str
    accuracy: float
    created_at: str
    is_active: bool

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    redis_connected: bool

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_connected = await redis_client.ping()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            models_loaded=len(model_manager.get_available_models()),
            redis_connected=redis_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make predictions using the ML model"""
    start_time = time.time()
    
    try:
        # Generate unique request ID
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Make prediction
        result = await prediction_service.predict(
            features=request.features,
            model_version=request.model_version,
            use_ab_testing=request.use_ab_testing,
            request_id=request_id
        )
        
        processing_time = time.time() - start_time
        
        # Log metrics in background
        background_tasks.add_task(
            metrics_collector.log_prediction,
            request_id=request_id,
            model_version=result["model_version"],
            processing_time=processing_time,
            features=request.features
        )
        
        return PredictionResponse(
            prediction=result["prediction"],
            model_version=result["model_version"],
            confidence=result.get("confidence"),
            processing_time=processing_time,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    try:
        models = model_manager.get_available_models()
        return [
            ModelInfo(
                version=model["version"],
                name=model["name"],
                accuracy=model["accuracy"],
                created_at=model["created_at"],
                is_active=model["is_active"]
            )
            for model in models
        ]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{version}/activate")
async def activate_model(version: str):
    """Activate a specific model version"""
    try:
        success = await model_manager.activate_model(version)
        if success:
            return {"message": f"Model {version} activated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Model version not found")
    except Exception as e:
        logger.error(f"Failed to activate model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    try:
        metrics = await metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ab-test/status")
async def ab_test_status():
    """Get A/B testing status"""
    try:
        status = await prediction_service.get_ab_test_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get A/B test status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ab-test/configure")
async def configure_ab_test(config: Dict[str, Any]):
    """Configure A/B testing parameters"""
    try:
        success = await prediction_service.configure_ab_test(config)
        if success:
            return {"message": "A/B test configuration updated"}
        else:
            raise HTTPException(status_code=400, detail="Invalid configuration")
    except Exception as e:
        logger.error(f"Failed to configure A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
