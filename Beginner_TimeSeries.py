# ============================================================
# ENERGY DEMAND FORECASTING (HOURLY) – ARIMA & SARIMA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("TimeSeries_TotalSolarGen_and_Load_IT_2016.csv")

df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

ts = df['Total_Load']

# ------------------------------------------------------------
# 2. EDA – TIME SERIES VISUALIZATION
# ------------------------------------------------------------
plt.figure(figsize=(14,5))
plt.plot(ts, color='steelblue')
plt.title("Hourly Energy Demand (Italy – 2016)")
plt.xlabel("Time")
plt.ylabel("Total Load")
plt.show()

# ------------------------------------------------------------
# 3. SEASONAL DECOMPOSITION
# ------------------------------------------------------------
decomp = seasonal_decompose(ts, model='additive', period=24)
decomp.plot()
plt.show()

# ------------------------------------------------------------
# 4. STATIONARITY CHECK (ADF TEST)
# ------------------------------------------------------------
adf_stat, p_value, _, _, _, _ = adfuller(ts)
print(f"ADF Statistic: {adf_stat}")
print(f"p-value: {p_value}")

# ------------------------------------------------------------
# 5. DIFFERENCING
# ------------------------------------------------------------
ts_diff = ts.diff().dropna()

# ------------------------------------------------------------
# 6. ACF & PACF
# ------------------------------------------------------------
plot_acf(ts_diff, lags=50)
plt.show()

plot_pacf(ts_diff, lags=50)
plt.show()

# ------------------------------------------------------------
# 7. TRAIN–TEST SPLIT (LAST 7 DAYS = 168 HOURS)
# ------------------------------------------------------------
train = ts[:-168]
test = ts[-168:]

# ------------------------------------------------------------
# 8. NAIVE BASELINE
# ------------------------------------------------------------
naive_forecast = np.repeat(train.iloc[-1], len(test))

# ------------------------------------------------------------
# 9. ARIMA MODEL
# ------------------------------------------------------------
arima_model = ARIMA(train, order=(1,1,1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=168)

# ------------------------------------------------------------
# 10. SARIMA MODEL (DAILY SEASONALITY)
# ------------------------------------------------------------
sarima_model = SARIMAX(
    train,
    order=(1,1,1),
    seasonal_order=(1,1,1,24),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=168)

# ------------------------------------------------------------
# 11. EVALUATION
# ------------------------------------------------------------
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

print("Naive:", evaluate(test, naive_forecast))
print("ARIMA:", evaluate(test, arima_forecast))
print("SARIMA:", evaluate(test, sarima_forecast))

# ------------------------------------------------------------
# 12. FORECAST VISUALIZATION
# ------------------------------------------------------------
plt.figure(figsize=(14,5))
plt.plot(test.index, test, label="Actual", color="black")
plt.plot(test.index, sarima_forecast, label="SARIMA Forecast", color="red")
plt.title("1-Week Hourly Energy Demand Forecast")
plt.xlabel("Time")
plt.ylabel("Total Load")
plt.legend()
plt.show()
