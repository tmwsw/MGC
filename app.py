import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from datetime import datetime

# Настройка страницы
st.set_page_config(page_title="Прогнозирование закупок", page_icon="📊", layout="wide")

# Заголовок приложения
st.title("📈 Прогнозирование закупок с помощью CatBoost")

# Загрузка модели
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboostmodel_Marin.cbm")  # Укажите путь к вашей модели
    return model

model = load_model()

# Загрузка исторических данных
@st.cache_data
def load_historical_data():
    df = pd.read_excel("combined_df.xlsx")
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(by="dt")
    return df

historical_prices = load_historical_data()

# Функция для расчета недель закупки
def calculate_weeks(price, predicted_price):
    if predicted_price > price * 1.04:
        return 1  # Цена растёт (+4%) → минимальная закупка
    elif predicted_price > price * 1.025:
        return 2  # Умеренный рост
    elif predicted_price > price * 1.01:
        return 3  # Незначительный рост
    elif predicted_price > price:
        return 4  # Незначительный рост (стабильный)
    elif predicted_price < price * 0.975:
        return 6  # Цена падает (-2%) → максимальная закупка
    else:
        return 5  # Стабильная цена → средний объём закупки

# Ввод даты и цены пользователем
date_input = st.date_input("Выберите дату для прогноза:", min_value=datetime.today())
price = st.number_input("Введите текущую цену на арматуру:", min_value=0.0, format="%.2f")

if st.button("🔮 Рассчитать прогноз", type="primary"):
    if price > 0:
        days_ahead = (date_input - datetime.today().date()).days
        predicted_price = np.expm1(model.predict([[price, days_ahead]])[0])
        weeks = calculate_weeks(price, predicted_price)
        
        st.subheader("📊 Результаты прогнозирования")
        st.write(f"📅 **Дата прогноза:** {date_input}")
        st.write(f"🔹 **Прогнозируемая цена:** {predicted_price:.2f}")
        st.write(f"📅 **Рекомендация по закупке:** {weeks} недель")
        
    else:
        st.warning("Введите корректную цену (больше 0)")