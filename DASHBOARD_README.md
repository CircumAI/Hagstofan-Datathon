# Iceland Export Prediction Dashboard

A beautiful, interactive web dashboard to visualize the Iceland export prediction model results.

## Features

- **Real-time Data Visualization**: Interactive charts showing export trends and exchange rates
- **Performance Metrics**: Display of model performance (MSE, MAE, RMSE, MAPE, Accuracy)
- **Model Architecture**: Detailed information about the BiLSTM + GRU ensemble
- **Feature Engineering**: Overview of all 22 engineered features
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Built with Bootstrap 5 and Chart.js

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Install required packages:

```bash
pip install flask pandas numpy
```

2. Run the dashboard:

```bash
python app.py
```

3. Open your browser and navigate to:

```
http://localhost:5000
```

## Dashboard Sections

### 1. Performance Metrics
- **MAE**: 5,893 ISK (8.9% improvement)
- **MSE**: 52.1M ISK² (14.9% improvement)
- **RMSE**: 7,221 ISK
- **MAPE**: 10.0%
- **Accuracy**: 90.0%

### 2. Data Visualizations
- **Export Trends**: Complete time series of Iceland's exports (2011-2025)
- **Exchange Rates**: EUR and USD rates over the last 24 months
- **Model Results**: Predictions vs actual values with training curves

### 3. Model Architecture
- **BiLSTM Component**: Bidirectional LSTM (128→64) + LSTM (32)
- **GRU Component**: GRU (128→64→32)
- **Ensemble**: Weighted averaging (BiLSTM 60%, GRU 40%)
- **22 Engineered Features**: Lags, rolling stats, EMAs, interactions, etc.

### 4. Training Configuration
- Huber loss function for robustness
- Gradient clipping (clipnorm=1.0)
- Adam optimizer with learning rate decay
- Early stopping on validation MAE
- L2 regularization and dropout

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: Bootstrap 5, Chart.js
- **Data Processing**: Pandas, NumPy
- **Charts**: Chart.js for interactive visualizations

## API Endpoints

The dashboard provides REST API endpoints:

- `GET /api/export-data` - Historical export data
- `GET /api/exchange-rates` - Exchange rate data
- `GET /api/metrics` - Model performance metrics
- `GET /api/model-info` - Model architecture details

## Customization

### Changing Colors
Edit `static/css/style.css` to modify the color scheme:

```css
:root {
    --primary-color: #0d6efd;
    --success-color: #198754;
    ...
}
```

### Adding New Charts
Add new chart functions in `static/js/dashboard.js` and call them in the `DOMContentLoaded` event.

### Modifying Metrics
Update the metrics in `app.py` in the `get_model_metrics()` function.

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Screenshots

The dashboard includes:
- Clean, modern interface with gradient hero section
- Interactive hover effects on metric cards
- Responsive charts that adapt to screen size
- Professional typography and spacing

## License

This dashboard is part of the Iceland Export Prediction project.

## Credits

Built for the Hagstofan Datathon 2025
