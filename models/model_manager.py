"""
Model Manager for handling ML model loading, versioning, and lifecycle
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from utils.config import Settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML model loading, versioning, and lifecycle"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: Dict[str, Dict[str, Any]] = {}
        self.active_model_version: Optional[str] = None
        self.models_directory = Path(settings.models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
    async def load_models(self):
        """Load all available models from disk"""
        logger.info("Loading models from disk...")
        
        # Create sample models if none exist
        if not any(self.models_directory.glob("*.pkl")):
            await self._create_sample_models()
        
        # Load existing models
        for model_file in self.models_directory.glob("*.pkl"):
            try:
                await self._load_model_from_file(model_file)
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
        
        # Set default active model
        if not self.active_model_version and self.models:
            self.active_model_version = max(self.models.keys())
            logger.info(f"Set active model to: {self.active_model_version}")
    
    async def _create_sample_models(self):
        """Create sample models for demonstration"""
        logger.info("Creating sample models...")
        
        # Generate sample data
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create different model versions
        models_to_create = [
            ("v1.0", RandomForestClassifier(n_estimators=50, random_state=42)),
            ("v2.0", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("v3.0", LogisticRegression(random_state=42))
        ]
        
        for version, model in models_to_create:
            # Train model
            model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            model_path = self.models_directory / f"model_{version}.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                "version": version,
                "name": f"Model {version}",
                "accuracy": float(accuracy),
                "created_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "features_count": X.shape[1],
                "training_samples": X_train.shape[0]
            }
            
            metadata_path = self.models_directory / f"model_{version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created model {version} with accuracy: {accuracy:.4f}")
    
    async def _load_model_from_file(self, model_file: Path):
        """Load a single model from file"""
        version = model_file.stem.replace("model_", "")
        
        try:
            # Load model
            model = joblib.load(model_file)
            
            # Load metadata
            metadata_file = model_file.parent / f"model_{version}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Create default metadata
                metadata = {
                    "version": version,
                    "name": f"Model {version}",
                    "accuracy": 0.0,
                    "created_at": datetime.now().isoformat(),
                    "model_type": type(model).__name__,
                    "features_count": getattr(model, 'n_features_in_', 0),
                    "training_samples": 0
                }
            
            # Store model and metadata
            self.models[version] = {
                "model": model,
                "metadata": metadata,
                "is_active": False,
                "load_time": datetime.now().isoformat()
            }
            
            logger.info(f"Loaded model {version}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")
            raise
    
    def get_model(self, version: Optional[str] = None) -> Optional[Any]:
        """Get a specific model or the active model"""
        if version is None:
            version = self.active_model_version
        
        if version and version in self.models:
            return self.models[version]["model"]
        
        return None
    
    def get_model_metadata(self, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model"""
        if version is None:
            version = self.active_model_version
        
        if version and version in self.models:
            return self.models[version]["metadata"]
        
        return None
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models with their metadata"""
        models_list = []
        
        for version, model_data in self.models.items():
            metadata = model_data["metadata"].copy()
            metadata["is_active"] = (version == self.active_model_version)
            models_list.append(metadata)
        
        # Sort by version
        models_list.sort(key=lambda x: x["version"], reverse=True)
        return models_list
    
    async def activate_model(self, version: str) -> bool:
        """Activate a specific model version"""
        if version not in self.models:
            logger.error(f"Model version {version} not found")
            return False
        
        # Deactivate current active model
        if self.active_model_version:
            self.models[self.active_model_version]["is_active"] = False
        
        # Activate new model
        self.active_model_version = version
        self.models[version]["is_active"] = True
        
        logger.info(f"Activated model version: {version}")
        return True
    
    async def add_model(self, version: str, model: Any, metadata: Dict[str, Any]) -> bool:
        """Add a new model to the manager"""
        try:
            # Save model to disk
            model_path = self.models_directory / f"model_{version}.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata_path = self.models_directory / f"model_{version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to memory
            self.models[version] = {
                "model": model,
                "metadata": metadata,
                "is_active": False,
                "load_time": datetime.now().isoformat()
            }
            
            logger.info(f"Added new model version: {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model {version}: {e}")
            return False
    
    async def remove_model(self, version: str) -> bool:
        """Remove a model version"""
        if version not in self.models:
            return False
        
        try:
            # Remove from memory
            del self.models[version]
            
            # Remove files
            model_path = self.models_directory / f"model_{version}.pkl"
            metadata_path = self.models_directory / f"model_{version}_metadata.json"
            
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Update active model if necessary
            if self.active_model_version == version:
                if self.models:
                    self.active_model_version = max(self.models.keys())
                else:
                    self.active_model_version = None
            
            logger.info(f"Removed model version: {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model {version}: {e}")
            return False
    
    def get_active_model_version(self) -> Optional[str]:
        """Get the currently active model version"""
        return self.active_model_version
    
    async def predict(self, features: List[float], version: Optional[str] = None) -> Dict[str, Any]:
        """Make a prediction using the specified or active model"""
        model = self.get_model(version)
        if model is None:
            raise ValueError(f"Model not found: {version or 'active'}")
        
        # Convert features to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            confidence = float(max(probabilities))
        
        return {
            "prediction": float(prediction),
            "confidence": confidence,
            "model_version": version or self.active_model_version
        }
