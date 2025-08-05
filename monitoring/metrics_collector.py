"""
Metrics Collector for monitoring ML model performance and system metrics
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import psutil
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages performance metrics for the ML API"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.metrics_buffer = defaultdict(deque)
        self.system_metrics = {}
        self.prediction_metrics = defaultdict(list)
        self.model_performance = defaultdict(dict)
        
        # Start background metrics collection
        asyncio.create_task(self._collect_system_metrics())
    
    async def log_prediction(
        self,
        request_id: str,
        model_version: str,
        processing_time: float,
        features: List[float],
        prediction: Optional[float] = None,
        confidence: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Log a prediction event with metrics"""
        
        timestamp = datetime.now()
        
        prediction_log = {
            "request_id": request_id,
            "model_version": model_version,
            "processing_time": processing_time,
            "timestamp": timestamp.isoformat(),
            "features_count": len(features),
            "prediction": prediction,
            "confidence": confidence,
            "error": error,
            "success": error is None
        }
        
        # Store in memory buffer
        self.prediction_metrics[model_version].append(prediction_log)
        
        # Keep only last 1000 predictions per model
        if len(self.prediction_metrics[model_version]) > 1000:
            self.prediction_metrics[model_version] = self.prediction_metrics[model_version][-1000:]
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    f"predictions:{model_version}",
                    json.dumps(prediction_log, default=str)
                )
                # Keep only last 10000 predictions per model in Redis
                await self.redis_client.ltrim(f"predictions:{model_version}", 0, 9999)
                
                # Update aggregated metrics
                await self._update_aggregated_metrics(model_version, prediction_log)
                
            except Exception as e:
                logger.warning(f"Failed to store prediction metrics in Redis: {e}")
    
    async def _update_aggregated_metrics(self, model_version: str, prediction_log: Dict[str, Any]) -> None:
        """Update aggregated metrics for a model"""
        try:
            # Get current aggregated metrics
            metrics_key = f"metrics:aggregated:{model_version}"
            current_metrics = await self.redis_client.get(metrics_key)
            
            if current_metrics:
                metrics = json.loads(current_metrics)
            else:
                metrics = {
                    "total_predictions": 0,
                    "successful_predictions": 0,
                    "failed_predictions": 0,
                    "avg_processing_time": 0.0,
                    "min_processing_time": float('inf'),
                    "max_processing_time": 0.0,
                    "avg_confidence": 0.0,
                    "last_updated": None
                }
            
            # Update metrics
            metrics["total_predictions"] += 1
            
            if prediction_log["success"]:
                metrics["successful_predictions"] += 1
                
                # Update processing time metrics
                processing_time = prediction_log["processing_time"]
                metrics["avg_processing_time"] = (
                    (metrics["avg_processing_time"] * (metrics["total_predictions"] - 1) + processing_time) /
                    metrics["total_predictions"]
                )
                metrics["min_processing_time"] = min(metrics["min_processing_time"], processing_time)
                metrics["max_processing_time"] = max(metrics["max_processing_time"], processing_time)
                
                # Update confidence metrics
                if prediction_log["confidence"] is not None:
                    if metrics["avg_confidence"] == 0.0:
                        metrics["avg_confidence"] = prediction_log["confidence"]
                    else:
                        metrics["avg_confidence"] = (
                            (metrics["avg_confidence"] * (metrics["successful_predictions"] - 1) + 
                             prediction_log["confidence"]) / metrics["successful_predictions"]
                        )
            else:
                metrics["failed_predictions"] += 1
            
            metrics["last_updated"] = datetime.now().isoformat()
            
            # Store updated metrics
            await self.redis_client.setex(
                metrics_key,
                86400,  # 24 hours TTL
                json.dumps(metrics, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Failed to update aggregated metrics: {e}")
    
    async def _collect_system_metrics(self) -> None:
        """Continuously collect system metrics"""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                system_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3)
                }
                
                # Store in memory
                self.system_metrics = system_metrics
                
                # Store in Redis if available
                if self.redis_client:
                    await self.redis_client.lpush(
                        "system_metrics",
                        json.dumps(system_metrics)
                    )
                    # Keep only last 1440 entries (24 hours at 1-minute intervals)
                    await self.redis_client.ltrim("system_metrics", 0, 1439)
                
                # Wait for 60 seconds before next collection
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                await asyncio.sleep(60)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        try:
            metrics_summary = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": self.system_metrics,
                "model_metrics": {},
                "api_metrics": await self._get_api_metrics()
            }
            
            # Get metrics for each model
            if self.redis_client:
                # Get all model versions that have metrics
                model_keys = []
                async for key in self.redis_client.scan_iter(match="metrics:aggregated:*"):
                    model_keys.append(key)
                
                for key in model_keys:
                    model_version = key.split(":")[-1]
                    metrics_data = await self.redis_client.get(key)
                    if metrics_data:
                        model_metrics = json.loads(metrics_data)
                        
                        # Add recent prediction trends
                        recent_predictions = await self._get_recent_predictions(model_version)
                        model_metrics["recent_trends"] = recent_predictions
                        
                        metrics_summary["model_metrics"][model_version] = model_metrics
            else:
                # Use in-memory metrics
                for model_version, predictions in self.prediction_metrics.items():
                    if predictions:
                        model_metrics = self._calculate_model_metrics(predictions)
                        metrics_summary["model_metrics"][model_version] = model_metrics
            
            return metrics_summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_recent_predictions(self, model_version: str, hours: int = 24) -> Dict[str, Any]:
        """Get recent prediction trends for a model"""
        try:
            # Get predictions from the last 24 hours
            predictions = await self.redis_client.lrange(f"predictions:{model_version}", 0, -1)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_predictions = []
            
            for pred_str in predictions:
                try:
                    pred = json.loads(pred_str)
                    pred_time = datetime.fromisoformat(pred["timestamp"])
                    if pred_time >= cutoff_time:
                        recent_predictions.append(pred)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            if not recent_predictions:
                return {"count": 0, "trends": {}}
            
            # Calculate hourly trends
            hourly_counts = defaultdict(int)
            hourly_avg_time = defaultdict(list)
            hourly_success_rate = defaultdict(list)
            
            for pred in recent_predictions:
                hour = datetime.fromisoformat(pred["timestamp"]).hour
                hourly_counts[hour] += 1
                hourly_avg_time[hour].append(pred["processing_time"])
                hourly_success_rate[hour].append(1 if pred["success"] else 0)
            
            trends = {}
            for hour in range(24):
                if hour in hourly_counts:
                    avg_time = sum(hourly_avg_time[hour]) / len(hourly_avg_time[hour])
                    success_rate = sum(hourly_success_rate[hour]) / len(hourly_success_rate[hour])
                    trends[str(hour)] = {
                        "count": hourly_counts[hour],
                        "avg_processing_time": avg_time,
                        "success_rate": success_rate
                    }
                else:
                    trends[str(hour)] = {
                        "count": 0,
                        "avg_processing_time": 0,
                        "success_rate": 0
                    }
            
            return {
                "count": len(recent_predictions),
                "trends": trends
            }
            
        except Exception as e:
            logger.warning(f"Failed to get recent predictions for {model_version}: {e}")
            return {"count": 0, "trends": {}, "error": str(e)}
    
    def _calculate_model_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics from in-memory predictions"""
        if not predictions:
            return {}
        
        total_predictions = len(predictions)
        successful_predictions = sum(1 for p in predictions if p["success"])
        failed_predictions = total_predictions - successful_predictions
        
        processing_times = [p["processing_time"] for p in predictions if p["success"]]
        confidences = [p["confidence"] for p in predictions if p["success"] and p["confidence"] is not None]
        
        metrics = {
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "success_rate": successful_predictions / total_predictions if total_predictions > 0 else 0,
            "last_updated": datetime.now().isoformat()
        }
        
        if processing_times:
            metrics.update({
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times)
            })
        
        if confidences:
            metrics["avg_confidence"] = sum(confidences) / len(confidences)
        
        return metrics
    
    async def _get_api_metrics(self) -> Dict[str, Any]:
        """Get API-level metrics"""
        try:
            api_metrics = {
                "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
                "total_requests": 0,
                "active_connections": 0,
                "error_rate": 0.0
            }
            
            if self.redis_client:
                # Get total requests across all models
                model_keys = []
                async for key in self.redis_client.scan_iter(match="metrics:aggregated:*"):
                    model_keys.append(key)
                
                total_requests = 0
                total_errors = 0
                
                for key in model_keys:
                    metrics_data = await self.redis_client.get(key)
                    if metrics_data:
                        model_metrics = json.loads(metrics_data)
                        total_requests += model_metrics.get("total_predictions", 0)
                        total_errors += model_metrics.get("failed_predictions", 0)
                
                api_metrics["total_requests"] = total_requests
                api_metrics["error_rate"] = total_errors / total_requests if total_requests > 0 else 0.0
            
            return api_metrics
            
        except Exception as e:
            logger.warning(f"Failed to get API metrics: {e}")
            return {"error": str(e)}
    
    async def get_model_comparison(self) -> Dict[str, Any]:
        """Get comparative metrics between models"""
        try:
            comparison = {
                "timestamp": datetime.now().isoformat(),
                "models": {}
            }
            
            if self.redis_client:
                # Get metrics for all models
                model_keys = []
                async for key in self.redis_client.scan_iter(match="metrics:aggregated:*"):
                    model_keys.append(key)
                
                for key in model_keys:
                    model_version = key.split(":")[-1]
                    metrics_data = await self.redis_client.get(key)
                    if metrics_data:
                        model_metrics = json.loads(metrics_data)
                        comparison["models"][model_version] = {
                            "success_rate": (
                                model_metrics.get("successful_predictions", 0) /
                                model_metrics.get("total_predictions", 1)
                            ),
                            "avg_processing_time": model_metrics.get("avg_processing_time", 0),
                            "avg_confidence": model_metrics.get("avg_confidence", 0),
                            "total_predictions": model_metrics.get("total_predictions", 0)
                        }
            
            # Calculate rankings
            if comparison["models"]:
                # Rank by success rate
                success_ranking = sorted(
                    comparison["models"].items(),
                    key=lambda x: x[1]["success_rate"],
                    reverse=True
                )
                
                # Rank by processing time (lower is better)
                speed_ranking = sorted(
                    comparison["models"].items(),
                    key=lambda x: x[1]["avg_processing_time"]
                )
                
                comparison["rankings"] = {
                    "by_success_rate": [model for model, _ in success_ranking],
                    "by_speed": [model for model, _ in speed_ranking]
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to get model comparison: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def set_start_time(self, start_time: float) -> None:
        """Set the application start time for uptime calculation"""
        self._start_time = start_time
