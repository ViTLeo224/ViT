# Завантаження бібліотек
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_excel('C:\\Users\\user\\dataset.xlsx')

# Перетворення часових даних
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = data[data['Value'].notnull() & (data['Value'] > 0)]

# Агрегація по днях
data['Date'] = data['Timestamp'].dt.date
daily_data = data.groupby('Date')['Value'].sum().reset_index()
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
daily_data.set_index('Date', inplace=True)

# Візуалізація даних
plt.figure(figsize=(12, 6))
plt.plot(daily_data.index, daily_data['Value'], label='Щоденні дані', color='blue')
plt.title('Агреговані щоденні дані')
plt.xlabel('Дата')
plt.ylabel('Сума Value за день')
plt.grid()
plt.legend()
plt.show()

# ARIMA-модель
model = ARIMA(daily_data['Value'], order=(3, 1, 3), seasonal_order=(1, 1, 1, 90))
fitted_model = model.fit()

# Підсумки моделі
print(fitted_model.summary())

# Прогноз на 6 місяців вперед (180 днів)
forecast_steps = 90
forecast = fitted_model.forecast(steps=forecast_steps)

# Побудова графіка прогнозу
plt.figure(figsize=(12, 6))
plt.plot(daily_data.index, daily_data['Value'], label='Історичні дані', color='blue')
forecast_index = pd.date_range(daily_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
plt.plot(forecast_index, forecast, label='Прогноз', color='orange')
plt.title('Прогноз на 6 місяців вперед (ARIMA)')
plt.xlabel('Дата')
plt.ylabel('Сума Value за день')
plt.legend()
plt.grid()
plt.show()

# Результати прогнозу
forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Predicted Value': forecast
})
print("Прогноз на наступні 6 місяців:")
print(forecast_df.head(10))  # Перші 10 значень
