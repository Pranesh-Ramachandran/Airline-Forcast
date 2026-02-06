# Airline Passenger Demand Forecasting with ARIMA

## Problem Statement
An airline company seeks to forecast monthly passenger demand to optimize aircraft allocation, workforce planning, and route scheduling. This project implements a complete ARIMA forecasting pipeline to handle non-stationary time series data.

## System Requirements
- Python 3.7+
- Statsmodels
- Pandas
- Matplotlib
- NumPy
- Scikit-learn

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python airline_forecast.py
```

## Pipeline Components

1. **Data Loading**: Loads historical airline passenger data
2. **Stationarity Testing**: Uses Augmented Dickey-Fuller (ADF) test
3. **Differencing**: Applies first and second-order differencing
4. **ACF/PACF Analysis**: Identifies optimal ARIMA parameters
5. **Model Fitting**: Trains ARIMA(2,1,2) model
6. **Evaluation**: Calculates RMSE, MAE, and MAPE
7. **Forecasting**: Generates future demand predictions

## Output
- `stationarity_analysis.png`: Visualization of differencing
- `acf_pacf.png`: ACF and PACF plots
- `forecast_results.png`: Model predictions vs actual
- `future_forecast.png`: 12-month future forecast

## Model Performance
The ARIMA model addresses non-stationarity through differencing (d=1) and captures temporal patterns using autoregressive (p=2) and moving average (q=2) components.
