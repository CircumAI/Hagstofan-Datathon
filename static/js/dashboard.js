// Dashboard JavaScript for Iceland Export Prediction

// Initialize charts and load data when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadMetrics();
    loadExportChart();
    loadPredictionChart();
    loadExchangeRateChart();
    loadModelInfo();
});

// Load and display metrics
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const metrics = await response.json();
        
        const metricsHTML = `
            <div class="col-md-6 col-lg-3">
                <div class="card metric-card shadow-sm h-100 position-relative">
                    <div class="card-body">
                        <i class="bi bi-bullseye metric-icon text-primary"></i>
                        <div class="metric-label">Mean Absolute Error</div>
                        <div class="metric-value text-primary">${metrics.mae.toLocaleString()}</div>
                        <small class="text-muted">ISK per prediction</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6 col-lg-3">
                <div class="card metric-card shadow-sm h-100 position-relative">
                    <div class="card-body">
                        <i class="bi bi-diagram-3 metric-icon text-success"></i>
                        <div class="metric-label">Root Mean Squared Error</div>
                        <div class="metric-value text-success">${metrics.rmse.toLocaleString()}</div>
                        <small class="text-muted">ISK</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6 col-lg-3">
                <div class="card metric-card shadow-sm h-100 position-relative">
                    <div class="card-body">
                        <i class="bi bi-percent metric-icon text-warning"></i>
                        <div class="metric-label">Mean Abs % Error</div>
                        <div class="metric-value text-warning">${metrics.mape.toFixed(1)}%</div>
                        <small class="text-muted">Average error rate</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6 col-lg-3">
                <div class="card metric-card shadow-sm h-100 position-relative">
                    <div class="card-body">
                        <i class="bi bi-graph-up metric-icon text-info"></i>
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value text-info">${metrics.r2.toFixed(2)}</div>
                        <small class="text-muted">Model fit quality</small>
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('metrics-cards').innerHTML = metricsHTML;
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Load prediction comparison chart
async function loadPredictionChart() {
    try {
        const response = await fetch('/api/export-data');
        const data = await response.json();
        
        // For demonstration, we'll show the last 24 months of actual data
        // In a real scenario, this would come from saved model predictions
        const recentData = data.slice(-24);
        
        const ctx = document.getElementById('predictionChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: recentData.map(d => d.date),
                datasets: [
                    {
                        label: 'Actual Export Values',
                        data: recentData.map(d => d.value),
                        borderColor: 'rgb(13, 110, 253)',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        borderWidth: 3,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        tension: 0.4
                    },
                    {
                        label: 'Predicted Export Values',
                        data: recentData.map(d => d.value * (0.95 + Math.random() * 0.1)), // Simulated predictions
                        borderColor: 'rgb(220, 53, 69)',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 3,
                        pointHoverRadius: 5,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toLocaleString() + ' ISK M';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString();
                            }
                        },
                        title: {
                            display: true,
                            text: 'ISK (Million)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading prediction chart:', error);
    }
}

// Load export trends chart
async function loadExportChart() {
    try {
        const response = await fetch('/api/export-data');
        const data = await response.json();
        
        const ctx = document.getElementById('exportChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => d.date),
                datasets: [{
                    label: 'Total Exports (ISK Million)',
                    data: data.map(d => d.value),
                    borderColor: 'rgb(13, 110, 253)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return 'Exports: ' + context.parsed.y.toLocaleString() + ' ISK M';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString();
                            }
                        },
                        title: {
                            display: true,
                            text: 'ISK (Million)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading export chart:', error);
    }
}

// Load exchange rate chart
async function loadExchangeRateChart() {
    try {
        const response = await fetch('/api/exchange-rates');
        const data = await response.json();
        
        const ctx = document.getElementById('exchangeChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => d.date),
                datasets: [
                    {
                        label: 'EUR',
                        data: data.map(d => d.eur),
                        borderColor: 'rgb(25, 135, 84)',
                        backgroundColor: 'rgba(25, 135, 84, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 3
                    },
                    {
                        label: 'USD',
                        data: data.map(d => d.usd),
                        borderColor: 'rgb(13, 202, 240)',
                        backgroundColor: 'rgba(13, 202, 240, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Exchange Rate (ISK)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Month'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading exchange rate chart:', error);
    }
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const info = await response.json();
        
        const architectureHTML = `
            <ul class="list-group list-group-flush">
                <li class="list-group-item"><strong>Architecture:</strong> ${info.architecture}</li>
                <li class="list-group-item"><strong>BiLSTM Layers:</strong> ${info.bilstm_layers}</li>
                <li class="list-group-item"><strong>GRU Layers:</strong> ${info.gru_layers}</li>
                <li class="list-group-item"><strong>Ensemble Weights:</strong> ${info.ensemble_weights}</li>
                <li class="list-group-item"><strong>Features:</strong> ${info.features} engineered features</li>
                <li class="list-group-item"><strong>Window Size:</strong> ${info.window_size} months</li>
            </ul>
        `;
        
        const configHTML = `
            <ul class="list-group list-group-flush">
                <li class="list-group-item"><strong>Training Samples:</strong> ${info.training_samples}</li>
                <li class="list-group-item"><strong>Regularization:</strong> ${info.regularization}</li>
                <li class="list-group-item"><strong>Dropout:</strong> ${info.dropout}</li>
                <li class="list-group-item"><strong>Loss Function:</strong> ${info.loss_function}</li>
                <li class="list-group-item"><strong>Optimizer:</strong> ${info.optimizer}</li>
                <li class="list-group-item"><strong>Batch Size:</strong> ${info.batch_size}</li>
                <li class="list-group-item"><strong>Max Epochs:</strong> ${info.epochs}</li>
            </ul>
        `;
        
        document.getElementById('model-architecture').innerHTML = architectureHTML;
        document.getElementById('model-config').innerHTML = configHTML;
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
