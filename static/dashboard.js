/**
 * Dashboard JavaScript for ML Model Serving API
 * Handles UI interactions, API calls, and real-time updates
 */

class MLDashboard {
    constructor() {
        this.apiBaseUrl = '';
        this.refreshInterval = 30000; // 30 seconds
        this.charts = {};
        this.init();
    }

    async init() {
        await this.loadModels();
        await this.loadMetrics();
        await this.loadABTestingStatus();
        this.setupEventListeners();
        this.generateFeatureInputs();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Prediction form submission
        document.getElementById('prediction-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.makePrediction();
        });

        // Model activation buttons (will be added dynamically)
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('activate-model-btn')) {
                const version = e.target.dataset.version;
                this.activateModel(version);
            }
        });
    }

    generateFeatureInputs() {
        const featuresGrid = document.getElementById('features-grid');
        // Generate 10 feature inputs (matching our sample model)
        for (let i = 0; i < 10; i++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01';
            input.placeholder = `Feature ${i + 1}`;
            input.className = 'border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500';
            input.id = `feature-${i}`;
            input.value = (Math.random() * 2 - 1).toFixed(2); // Random value between -1 and 1
            featuresGrid.appendChild(input);
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/models');
            const models = await response.json();
            
            this.updateModelsDisplay(models);
            this.updateModelSelect(models);
            
            // Update active models count
            const activeCount = models.filter(m => m.is_active).length;
            document.getElementById('active-models-count').textContent = activeCount;
            
        } catch (error) {
            console.error('Failed to load models:', error);
            this.showError('Failed to load models');
        }
    }

    updateModelsDisplay(models) {
        const modelsList = document.getElementById('models-list');
        modelsList.innerHTML = '';

        models.forEach(model => {
            const modelCard = document.createElement('div');
            modelCard.className = 'border rounded-lg p-4 mb-4';
            
            const statusClass = model.is_active ? 'bg-green-100 border-green-300' : 'bg-gray-100 border-gray-300';
            modelCard.className += ` ${statusClass}`;

            modelCard.innerHTML = `
                <div class="flex justify-between items-start">
                    <div>
                        <h4 class="font-medium text-gray-900">${model.name}</h4>
                        <p class="text-sm text-gray-600">Version: ${model.version}</p>
                        <p class="text-sm text-gray-600">Accuracy: ${(model.accuracy * 100).toFixed(2)}%</p>
                        <p class="text-sm text-gray-600">Created: ${new Date(model.created_at).toLocaleDateString()}</p>
                    </div>
                    <div class="flex flex-col items-end">
                        ${model.is_active ? 
                            '<span class="bg-green-500 text-white px-2 py-1 rounded text-xs">Active</span>' :
                            `<button class="activate-model-btn bg-blue-500 text-white px-3 py-1 rounded text-xs hover:bg-blue-600" data-version="${model.version}">Activate</button>`
                        }
                    </div>
                </div>
            `;
            
            modelsList.appendChild(modelCard);
        });
    }

    updateModelSelect(models) {
        const modelSelect = document.getElementById('model-select');
        // Clear existing options except the first one
        while (modelSelect.children.length > 1) {
            modelSelect.removeChild(modelSelect.lastChild);
        }

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.version;
            option.textContent = `${model.name} (${model.version})`;
            if (model.is_active) {
                option.textContent += ' - Active';
            }
            modelSelect.appendChild(option);
        });
    }

    async loadMetrics() {
        try {
            const response = await fetch('/metrics');
            const metrics = await response.json();
            
            this.updateMetricsDisplay(metrics);
            this.updateChartsDisplay(metrics);
            
        } catch (error) {
            console.error('Failed to load metrics:', error);
            this.showError('Failed to load metrics');
        }
    }

    updateMetricsDisplay(metrics) {
        // Update quick stats
        let totalPredictions = 0;
        let totalSuccessful = 0;
        let avgProcessingTime = 0;
        let modelCount = 0;

        if (metrics.model_metrics) {
            Object.values(metrics.model_metrics).forEach(modelMetrics => {
                totalPredictions += modelMetrics.total_predictions || 0;
                totalSuccessful += modelMetrics.successful_predictions || 0;
                if (modelMetrics.avg_processing_time) {
                    avgProcessingTime += modelMetrics.avg_processing_time;
                    modelCount++;
                }
            });
        }

        document.getElementById('total-predictions').textContent = totalPredictions.toLocaleString();
        
        const successRate = totalPredictions > 0 ? (totalSuccessful / totalPredictions * 100) : 0;
        document.getElementById('success-rate').textContent = `${successRate.toFixed(1)}%`;
        
        const avgTime = modelCount > 0 ? (avgProcessingTime / modelCount) : 0;
        document.getElementById('avg-response-time').textContent = `${(avgTime * 1000).toFixed(0)}ms`;

        // Update system metrics
        if (metrics.system_metrics) {
            const sys = metrics.system_metrics;
            document.getElementById('cpu-usage').textContent = `${sys.cpu_percent?.toFixed(1) || 0}%`;
            document.getElementById('memory-usage').textContent = `${sys.memory_percent?.toFixed(1) || 0}%`;
            document.getElementById('disk-usage').textContent = `${sys.disk_percent?.toFixed(1) || 0}%`;
        }
    }

    updateChartsDisplay(metrics) {
        // Create or update trends chart
        const ctx = document.getElementById('trends-chart').getContext('2d');
        
        if (this.charts.trends) {
            this.charts.trends.destroy();
        }

        // Prepare data for trends chart
        const hours = Array.from({length: 24}, (_, i) => i);
        const datasets = [];

        if (metrics.model_metrics) {
            Object.entries(metrics.model_metrics).forEach(([modelVersion, modelMetrics], index) => {
                if (modelMetrics.recent_trends && modelMetrics.recent_trends.trends) {
                    const data = hours.map(hour => {
                        const trend = modelMetrics.recent_trends.trends[hour.toString()];
                        return trend ? trend.count : 0;
                    });

                    const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
                    const color = colors[index % colors.length];

                    datasets.push({
                        label: modelVersion,
                        data: data,
                        borderColor: color,
                        backgroundColor: color + '20',
                        tension: 0.4
                    });
                }
            });
        }

        this.charts.trends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours.map(h => `${h}:00`),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Predictions Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Hour of Day'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    async loadABTestingStatus() {
        try {
            const response = await fetch('/ab-test/status');
            const status = await response.json();
            
            this.updateABTestingDisplay(status);
            
        } catch (error) {
            console.error('Failed to load A/B testing status:', error);
            this.showError('Failed to load A/B testing status');
        }
    }

    updateABTestingDisplay(status) {
        const container = document.getElementById('ab-testing-status');
        
        if (!status.enabled) {
            container.innerHTML = '<p class="text-gray-500">A/B Testing is disabled</p>';
            return;
        }

        let html = `
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <span class="text-sm font-medium">Status:</span>
                    <span class="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Enabled</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-sm font-medium">Strategy:</span>
                    <span class="text-sm">${status.config?.strategy || 'random'}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-sm font-medium">Split Ratio:</span>
                    <span class="text-sm">${((status.config?.split_ratio || 0.5) * 100).toFixed(0)}%</span>
                </div>
        `;

        if (status.model_percentages && Object.keys(status.model_percentages).length > 0) {
            html += '<div class="mt-4"><h4 class="text-sm font-medium mb-2">Model Distribution:</h4>';
            Object.entries(status.model_percentages).forEach(([model, percentage]) => {
                html += `
                    <div class="flex items-center justify-between mb-1">
                        <span class="text-sm">${model}:</span>
                        <span class="text-sm font-medium">${percentage.toFixed(1)}%</span>
                    </div>
                `;
            });
            html += '</div>';
        }

        html += '</div>';
        container.innerHTML = html;
    }

    async makePrediction() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const resultDiv = document.getElementById('prediction-result');
        
        try {
            loadingOverlay.classList.remove('hidden');
            
            // Collect features
            const features = [];
            for (let i = 0; i < 10; i++) {
                const value = parseFloat(document.getElementById(`feature-${i}`).value) || 0;
                features.push(value);
            }

            // Prepare request
            const requestData = {
                features: features,
                use_ab_testing: document.getElementById('ab-testing-enabled').checked
            };

            const modelVersion = document.getElementById('model-select').value;
            if (modelVersion) {
                requestData.model_version = modelVersion;
            }

            // Make prediction
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            // Display result
            document.getElementById('result-prediction').textContent = result.prediction.toFixed(4);
            document.getElementById('result-model').textContent = result.model_version;
            document.getElementById('result-confidence').textContent = 
                result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : 'N/A';
            document.getElementById('result-time').textContent = `${(result.processing_time * 1000).toFixed(0)}ms`;
            document.getElementById('result-id').textContent = result.request_id;
            
            resultDiv.classList.remove('hidden');
            
            // Refresh metrics after prediction
            setTimeout(() => this.loadMetrics(), 1000);
            
        } catch (error) {
            console.error('Prediction failed:', error);
            this.showError(`Prediction failed: ${error.message}`);
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    async activateModel(version) {
        try {
            const response = await fetch(`/models/${version}/activate`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.showSuccess(result.message);
            
            // Refresh models display
            await this.loadModels();
            
        } catch (error) {
            console.error('Model activation failed:', error);
            this.showError(`Failed to activate model: ${error.message}`);
        }
    }

    async checkHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            
            const statusIndicator = document.getElementById('status-indicator');
            if (health.status === 'healthy') {
                statusIndicator.innerHTML = `
                    <div class="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                    <span>Online</span>
                `;
            } else {
                statusIndicator.innerHTML = `
                    <div class="w-3 h-3 bg-red-400 rounded-full mr-2"></div>
                    <span>Offline</span>
                `;
            }
            
        } catch (error) {
            console.error('Health check failed:', error);
            const statusIndicator = document.getElementById('status-indicator');
            statusIndicator.innerHTML = `
                <div class="w-3 h-3 bg-red-400 rounded-full mr-2"></div>
                <span>Offline</span>
            `;
        }
    }

    startAutoRefresh() {
        // Refresh metrics every 30 seconds
        setInterval(() => {
            this.loadMetrics();
            this.loadABTestingStatus();
            this.checkHealth();
        }, this.refreshInterval);

        // Initial health check
        this.checkHealth();
    }

    showError(message) {
        // Create a simple toast notification
        const toast = document.createElement('div');
        toast.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded shadow-lg z-50';
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 5000);
    }

    showSuccess(message) {
        // Create a simple toast notification
        const toast = document.createElement('div');
        toast.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded shadow-lg z-50';
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 3000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MLDashboard();
});
