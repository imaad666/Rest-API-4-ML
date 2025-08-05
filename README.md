# ML Model Serving API

A comprehensive Machine Learning model serving API built with FastAPI, featuring model versioning, A/B testing, async request processing, and performance monitoring.

## 🚀 Features

- **REST API for ML Predictions**: Fast and reliable prediction endpoints
- **Model Versioning**: Support for multiple model versions with easy switching
- **A/B Testing**: Built-in A/B testing framework with multiple strategies
- **Async Processing**: High-performance async request handling
- **Performance Monitoring**: Real-time metrics collection and analysis
- **Web Dashboard**: Interactive dashboard for model testing and monitoring
- **Redis Integration**: Caching and data persistence
- **Comprehensive Logging**: Detailed logging and error tracking

## 🏗️ Architecture

```
ml-model-api/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment configuration template
├── models/
│   ├── __init__.py
│   ├── model_manager.py   # Model loading and management
│   ├── prediction_service.py  # Prediction logic with A/B testing
│   └── artifacts/         # Stored model files (auto-created)
├── monitoring/
│   ├── __init__.py
│   └── metrics_collector.py  # Performance metrics collection
├── utils/
│   ├── __init__.py
│   └── config.py          # Configuration management
├── templates/
│   └── dashboard.html     # Web dashboard template
├── static/
│   └── dashboard.js       # Dashboard JavaScript
└── tests/
    ├── __init__.py
    └── test_api.py         # API test suite
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Redis Server
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-model-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Redis server**
   ```bash
   # On Windows with Redis installed
   redis-server
   
   # On Linux/Mac
   sudo systemctl start redis
   # or
   brew services start redis
   ```

5. **Configure environment (optional)**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📊 Dashboard

Access the interactive dashboard at `http://localhost:8000` to:

- Test model predictions with custom inputs
- Monitor real-time performance metrics
- Manage model versions
- View A/B testing statistics
- Monitor system resources

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  "model_version": "v1.0",  // optional
  "use_ab_testing": true    // optional, default: true
}
```

### List Models
```http
GET /models
```

### Activate Model
```http
POST /models/{version}/activate
```

### Get Metrics
```http
GET /metrics
```

### A/B Testing Status
```http
GET /ab-test/status
```

### Configure A/B Testing
```http
POST /ab-test/configure
Content-Type: application/json

{
  "enabled": true,
  "split_ratio": 0.5,
  "models": ["v1.0", "v2.0"],
  "strategy": "random"  // random, hash-based, or weighted
}
```

## 🧪 A/B Testing

The API supports three A/B testing strategies:

1. **Random**: Random assignment based on split ratio
2. **Hash-based**: Consistent assignment based on request identifier
3. **Weighted**: Assignment based on model performance metrics

## 📈 Monitoring

### Metrics Collected

- **Prediction Metrics**: Count, success rate, processing time, confidence
- **Model Performance**: Accuracy, response time, error rates
- **System Metrics**: CPU, memory, disk usage
- **A/B Testing**: Distribution, conversion rates

### Real-time Dashboard

The dashboard provides:
- Live performance charts
- Model comparison analytics
- System resource monitoring
- A/B test result visualization

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Redis Settings
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Settings
DEFAULT_MODEL_VERSION=v1.0
MAX_MODELS_IN_MEMORY=3

# A/B Testing
AB_TEST_ENABLED=true
AB_TEST_SPLIT_RATIO=0.5

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
PREDICTION_CACHE_TTL=300
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_api.py::test_prediction -v
```

## 📦 Model Management

### Adding New Models

1. Train your scikit-learn model
2. Save using joblib:
   ```python
   import joblib
   joblib.dump(model, 'models/artifacts/model_v2.0.pkl')
   ```
3. Create metadata file:
   ```json
   {
     "version": "v2.0",
     "name": "Model v2.0",
     "accuracy": 0.95,
     "created_at": "2024-01-01T00:00:00",
     "model_type": "RandomForestClassifier"
   }
   ```
4. Restart the API or use the management endpoints

### Model Versioning

- Models are automatically loaded on startup
- Support for multiple concurrent model versions
- Easy switching between model versions
- Rollback capabilities

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Considerations

1. **Use a production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Set up Redis cluster** for high availability

3. **Configure monitoring** with Prometheus/Grafana

4. **Set up load balancing** for multiple instances

5. **Enable API authentication** in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 API Documentation

Once running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## 🐛 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis server is running
   - Check Redis host/port configuration

2. **Model Loading Fails**
   - Verify model files exist in `models/artifacts/`
   - Check model file permissions

3. **High Memory Usage**
   - Reduce `MAX_MODELS_IN_MEMORY` setting
   - Monitor model sizes

4. **Slow Predictions**
   - Enable Redis caching
   - Check model complexity
   - Monitor system resources

### Logs

Application logs are available in the console output. For production, configure proper logging:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- scikit-learn for machine learning capabilities
- Redis for caching and data persistence
- Chart.js for dashboard visualizations

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test suite for examples

---

**Happy ML Serving! 🤖✨**
