import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_data():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    df.columns = ['Passengers']
    return df


def test_stationarity(series, title):
    result = adfuller(series.dropna())
    print(f'\n{title}')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Stationary: {"Yes" if result[1] < 0.05 else "No"}')
    return result[1] < 0.05


def plot_series(original, diff1, diff2):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    original.plot(ax=axes[0], title='Original Series')
    axes[0].set_ylabel('Passengers')
    
    diff1.plot(ax=axes[1], title='First Difference')
    axes[1].set_ylabel('Diff 1')
    
    diff2.plot(ax=axes[2], title='Second Difference')
    axes[2].set_ylabel('Diff 2')
    
    plt.tight_layout()
    plt.savefig('stationarity_analysis.png')
    plt.show()


def plot_acf_pacf(series):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=40, ax=axes[0])
    plot_pacf(series.dropna(), lags=40, ax=axes[1])
    plt.tight_layout()
    plt.savefig('acf_pacf.png')
    plt.show()


def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f'\nModel Performance:')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape:.2f}%')
    return rmse, mae, mape


def main():
    print("="*60)
    print("AIRLINE PASSENGER DEMAND FORECASTING WITH ARIMA")
    print("="*60)
    
    
    df = load_data()
    print(f'\nDataset shape: {df.shape}')
    print(f'Date range: {df.index[0]} to {df.index[-1]}')
    
    
    test_stationarity(df['Passengers'], 'Original Series - Stationarity Test')
    
    
    df['Diff1'] = df['Passengers'].diff()
    df['Diff2'] = df['Diff1'].diff()
    
    test_stationarity(df['Diff1'], 'First Difference - Stationarity Test')
    test_stationarity(df['Diff2'], 'Second Difference - Stationarity Test')
    
    
    plot_series(df['Passengers'], df['Diff1'], df['Diff2'])
    plot_acf_pacf(df['Diff1'])
    
   
    train_size = int(len(df) * 0.8)
    train, test = df['Passengers'][:train_size], df['Passengers'][train_size:]
    
    print(f'\nTrain size: {len(train)}, Test size: {len(test)}')
    
    
    print('\nFitting ARIMA(2,1,2) model...')
    model = ARIMA(train, order=(2, 1, 2))
    fitted_model = model.fit()
    print(fitted_model.summary())
    
    
    predictions = fitted_model.forecast(steps=len(test))
    
    
    evaluate_model(test.values, predictions)
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train', color='blue')
    plt.plot(test.index, test, label='Actual', color='green')
    plt.plot(test.index, predictions, label='Forecast', color='red', linestyle='--')
    plt.title('Airline Passenger Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_results.png')
    plt.show()
    
    
    print('\nGenerating 12-month future forecast...')
    future_model = ARIMA(df['Passengers'], order=(2, 1, 2))
    future_fitted = future_model.fit()
    future_forecast = future_fitted.forecast(steps=12)
    
    print('\nFuture 12-Month Forecast:')
    for i, val in enumerate(future_forecast, 1):
        print(f'Month {i}: {val:.0f} passengers')
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Passengers'], label='Historical', color='blue')
    future_dates = pd.date_range(start=df.index[-1], periods=13, freq='MS')[1:]
    plt.plot(future_dates, future_forecast, label='Future Forecast', color='red', linestyle='--', marker='o')
    plt.title('12-Month Future Passenger Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('future_forecast.png')
    plt.show()
    
    print('\n' + '='*60)
    print('Forecasting pipeline completed successfully!')
    print('='*60)

if __name__ == '__main__':
    main()
