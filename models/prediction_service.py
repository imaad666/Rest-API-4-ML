"""
Prediction Service for handling async predictions and A/B testing
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

import redis.asyncio as redis
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class PredictionService:
    """Handles prediction requests with A/B testing and caching"""
    
    def __init__(self, model_manager: ModelManager, redis_client: redis.Redis):
        self.model_manager = model_manager
        self.redis_client = redis_client
        self.ab_test_config = {
            "enabled": True,
            "split_ratio": 0.5,
            "models": ["v1.0", "v2.0"],
            "strategy": "random"  # random, hash-based, or weighted
        }
    
    async def predict(
        self,
        features: List[float],
        model_version: Optional[str] = None,
        use_ab_testing: bool = True,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a prediction with optional A/B testing"""
        
        # Determine which model to use
        if model_version is None and use_ab_testing and self.ab_test_config["enabled"]:
            model_version = await self._select_ab_test_model(features, request_id)
        elif model_version is None:
            model_version = self.model_manager.get_active_model_version()
        
        if model_version is None:
            raise ValueError("No model available for prediction")
        
        # Check cache first
        cache_key = self._generate_cache_key(features, model_version)
        cached_result = await self._get_cached_prediction(cache_key)
        if cached_result:
            logger.info(f"Returning cached prediction for model {model_version}")
            return cached_result
        
        # Make prediction
        try:
            result = await self.model_manager.predict(features, model_version)
            
            # Add metadata
            result.update({
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "cached": False
            })
            
            # Cache the result
            await self._cache_prediction(cache_key, result)
            
            # Log A/B test assignment if applicable
            if use_ab_testing and self.ab_test_config["enabled"]:
                await self._log_ab_test_assignment(request_id, model_version, features)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_version}: {e}")
            raise
    
    async def _select_ab_test_model(
        self,
        features: List[float],
        request_id: Optional[str] = None
    ) -> str:
        """Select model for A/B testing based on configured strategy"""
        
        available_models = self.ab_test_config.get("models", [])
        if len(available_models) < 2:
            # Fall back to active model if not enough models for A/B testing
            return self.model_manager.get_active_model_version()
        
        strategy = self.ab_test_config.get("strategy", "random")
        
        if strategy == "random":
            return await self._random_model_selection(available_models)
        elif strategy == "hash-based":
            return await self._hash_based_model_selection(available_models, request_id or str(features))
        elif strategy == "weighted":
            return await self._weighted_model_selection(available_models)
        else:
            return available_models[0]  # Default to first model
    
    async def _random_model_selection(self, models: List[str]) -> str:
        """Random model selection for A/B testing"""
        split_ratio = self.ab_test_config.get("split_ratio", 0.5)
        
        if random.random() < split_ratio:
            return models[0]
        else:
            return models[1] if len(models) > 1 else models[0]
    
    async def _hash_based_model_selection(self, models: List[str], identifier: str) -> str:
        """Hash-based model selection for consistent assignment"""
        hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
        split_ratio = self.ab_test_config.get("split_ratio", 0.5)
        
        # Use hash to determine assignment consistently
        if (hash_value % 100) / 100 < split_ratio:
            return models[0]
        else:
            return models[1] if len(models) > 1 else models[0]
    
    async def _weighted_model_selection(self, models: List[str]) -> str:
        """Weighted model selection based on performance metrics"""
        # Get model performance metrics
        weights = []
        for model_version in models:
            metadata = self.model_manager.get_model_metadata(model_version)
            if metadata:
                # Use accuracy as weight (could be extended to other metrics)
                weights.append(metadata.get("accuracy", 0.5))
            else:
                weights.append(0.5)  # Default weight
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(models)] * len(models)
        
        # Select model based on weights
        rand_value = random.random()
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if rand_value <= cumulative_weight:
                return models[i]
        
        return models[-1]  # Fallback to last model
    
    def _generate_cache_key(self, features: List[float], model_version: str) -> str:
        """Generate cache key for prediction"""
        features_str = ",".join(map(str, features))
        return f"prediction:{model_version}:{hashlib.md5(features_str.encode()).hexdigest()}"
    
    async def _get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction result"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                result = json.loads(cached_data)
                result["cached"] = True
                return result
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_prediction(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache prediction result"""
        try:
            # Cache for 5 minutes (300 seconds)
            await self.redis_client.setex(
                cache_key,
                300,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _log_ab_test_assignment(
        self,
        request_id: Optional[str],
        model_version: str,
        features: List[float]
    ) -> None:
        """Log A/B test assignment for analysis"""
        try:
            assignment_data = {
                "request_id": request_id,
                "model_version": model_version,
                "timestamp": datetime.now().isoformat(),
                "features_hash": hashlib.md5(str(features).encode()).hexdigest()
            }
            
            # Store in Redis list for later analysis
            await self.redis_client.lpush(
                "ab_test_assignments",
                json.dumps(assignment_data)
            )
            
            # Keep only last 10000 assignments
            await self.redis_client.ltrim("ab_test_assignments", 0, 9999)
            
        except Exception as e:
            logger.warning(f"A/B test logging failed: {e}")
    
    async def get_ab_test_status(self) -> Dict[str, Any]:
        """Get current A/B test status and statistics"""
        try:
            # Get recent assignments
            assignments = await self.redis_client.lrange("ab_test_assignments", 0, -1)
            
            # Parse assignments
            parsed_assignments = []
            for assignment in assignments:
                try:
                    parsed_assignments.append(json.loads(assignment))
                except json.JSONDecodeError:
                    continue
            
            # Calculate statistics
            total_assignments = len(parsed_assignments)
            model_counts = {}
            
            for assignment in parsed_assignments:
                model_version = assignment.get("model_version", "unknown")
                model_counts[model_version] = model_counts.get(model_version, 0) + 1
            
            # Calculate percentages
            model_percentages = {}
            if total_assignments > 0:
                for model, count in model_counts.items():
                    model_percentages[model] = (count / total_assignments) * 100
            
            return {
                "enabled": self.ab_test_config["enabled"],
                "config": self.ab_test_config,
                "total_assignments": total_assignments,
                "model_counts": model_counts,
                "model_percentages": model_percentages,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get A/B test status: {e}")
            return {
                "enabled": self.ab_test_config["enabled"],
                "config": self.ab_test_config,
                "error": str(e)
            }
    
    async def configure_ab_test(self, config: Dict[str, Any]) -> bool:
        """Configure A/B testing parameters"""
        try:
            # Validate configuration
            if "enabled" in config:
                self.ab_test_config["enabled"] = bool(config["enabled"])
            
            if "split_ratio" in config:
                split_ratio = float(config["split_ratio"])
                if 0 <= split_ratio <= 1:
                    self.ab_test_config["split_ratio"] = split_ratio
                else:
                    raise ValueError("split_ratio must be between 0 and 1")
            
            if "models" in config:
                models = config["models"]
                if isinstance(models, list) and len(models) >= 2:
                    # Verify models exist
                    available_models = [m["version"] for m in self.model_manager.get_available_models()]
                    for model in models:
                        if model not in available_models:
                            raise ValueError(f"Model {model} not found")
                    self.ab_test_config["models"] = models
                else:
                    raise ValueError("models must be a list with at least 2 models")
            
            if "strategy" in config:
                strategy = config["strategy"]
                if strategy in ["random", "hash-based", "weighted"]:
                    self.ab_test_config["strategy"] = strategy
                else:
                    raise ValueError("strategy must be one of: random, hash-based, weighted")
            
            logger.info(f"A/B test configuration updated: {self.ab_test_config}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure A/B test: {e}")
            return False
    
    async def batch_predict(
        self,
        batch_features: List[List[float]],
        model_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple predictions concurrently"""
        tasks = []
        
        for i, features in enumerate(batch_features):
            request_id = f"batch_{int(time.time())}_{i}"
            task = self.predict(
                features=features,
                model_version=model_version,
                use_ab_testing=False,  # Disable A/B testing for batch
                request_id=request_id
            )
            tasks.append(task)
        
        # Execute all predictions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "request_id": f"batch_{int(time.time())}_{i}",
                    "index": i
                })
            else:
                processed_results.append(result)
        
        return processed_results
