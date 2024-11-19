import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Заголовок додатку
st.title("Прогнозування на основі ARIMA-моделі")

# Завантаження файлу
uploaded_file = st.file_uploader("Завантажте ваш файл CSV або Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Завантаження даних
    if uploaded_file.name.endswith('xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    # Перевірка на наявність колонок
    if 'Timestamp' not in data.columns or 'Value' not in data.columns:
        st.error("Дані повинні містити колонки 'Timestamp' і 'Value'")
    else:
        # Перетворення часових даних
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data = data[data['Value'].notnull() & (data['Value'] > 0)]

        # Використання щоденних даних без агрегації
        daily_data = data.groupby('Timestamp').sum()  # Агрегуємо дані по днях

        # ARIMA-модель
        model = ARIMA(daily_data['Value'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 30))  # Сезонність ~1 місяць
        fitted_model = model.fit()

        # Підсумки моделі
        st.subheader('Підсумки ARIMA-моделі')
        st.text(fitted_model.summary())

        # Прогноз на 6 місяців вперед (~180 днів)
        forecast_steps = 180
        forecast = fitted_model.forecast(steps=forecast_steps)

        # Побудова графіка прогнозу
        forecast_index = pd.date_range(daily_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

        st.subheader('Прогноз на 6 місяців вперед')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_data.index, daily_data['Value'], label='Історичні дані', color='blue')
        ax.plot(forecast_index, forecast, label='Прогноз', color='orange')
        ax.set_title('Прогноз на 6 місяців вперед (ARIMA, щоденні дані)')
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
        st.write(forecast_df.head(10))  # Виведемо перші 10 значень прогнозу
