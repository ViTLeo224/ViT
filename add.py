import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Заголовок додатку
st.title("Прогнозування за допомогою ARIMA-моделі")

# Завантаження файлу
uploaded_file = st.file_uploader("Завантажте ваш файл CSV", type="csv")

if uploaded_file is not None:
    # Завантаження даних
    data = pd.read_csv(uploaded_file)

    # Перетворення часових даних
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data[data['Value'].notnull() & (data['Value'] > 0)]

    # Агрегація по днях
    data['Date'] = data['Timestamp'].dt.date
    daily_data = data.groupby('Date')['Value'].sum().reset_index()
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_data.set_index('Date', inplace=True)

    # Візуалізація даних
    st.subheader('Агреговані щоденні дані')
    st.line_chart(daily_data['Value'])

    # ARIMA-модель
    model = ARIMA(daily_data['Value'], order=(3, 1, 3), seasonal_order=(1, 1, 1, 90))
    fitted_model = model.fit()

    # Підсумки моделі
    st.subheader('Підсумки ARIMA-моделі')
    st.text(fitted_model.summary())

    # Прогноз на 6 місяців вперед (180 днів)
    forecast_steps = 90
    forecast = fitted_model.forecast(steps=forecast_steps)

    # Побудова графіка прогнозу
    forecast_index = pd.date_range(daily_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

    st.subheader('Прогноз на 6 місяців вперед')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_data.index, daily_data['Value'], label='Історичні дані', color='blue')
    ax.plot(forecast_index, forecast, label='Прогноз', color='orange')
    ax.set_title('Прогноз на 6 місяців вперед (ARIMA)')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Сума Value за день')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Результати прогнозу
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Predicted Value': forecast
    })
    st.subheader('Результати прогнозу')
    st.write(forecast_df.head(10))
