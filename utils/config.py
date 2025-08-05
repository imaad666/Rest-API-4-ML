"""
Configuration settings for the ML API
"""

import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    api_title: str = "ML Model Serving API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # Model Settings
    models_directory: str = "models/artifacts"
    default_model_version: str = "v1.0"
    max_models_in_memory: int = 3
    
    # A/B Testing Settings
    ab_test_enabled: bool = True
    ab_test_split_ratio: float = 0.5  # 50/50 split
    ab_test_models: List[str] = ["v1.0", "v2.0"]
    
    # Performance Settings
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    prediction_cache_ttl: int = 300  # 5 minutes
    
    # Monitoring Settings
    metrics_retention_days: int = 30
    enable_detailed_logging: bool = True
    
    # Security Settings
    api_key_required: bool = False
    allowed_origins: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
