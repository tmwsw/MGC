import streamlit as st
import pandas as pd
import numpy as np
import io
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(page_title="Прогнозирование закупок", page_icon="📊", layout="wide")

# Заголовок приложения
st.title("📈 Прогнозирование закупок с помощью CatBoost")
st.markdown("""
    Это приложение позволяет загрузить данные в формате Excel и выполнить прогнозирование 
    с использованием CatBoost-модели. Результаты включают прогнозируемую цену и рекомендации 
    по количеству недель для закупки.
""")

# Загрузка модели
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboostmodel_Marin.cbm")  # Укажите путь к вашей модели
    return model

model = load_model()

# Функция для обработки данных и предсказаний
def process_data(df):
    # Преобразуем дату
    df["dt"] = pd.to_datetime(df["dt"], dayfirst=True)
    df.rename(columns={'Цена на арматуру': 'Price'}, inplace=True, errors='ignore')

    # Создаём признаки
    if 'Price' in df.columns:
        df["Price_source"] = df["Price"].shift(1)

        # Заполняем NaN
        df["Price_source"] = df["Price_source"].interpolate(method='linear')
        df["Price_source"] = df["Price_source"].fillna(df["Price_source"].mean())

    # Подготовка данных для предсказания
    X_test = df[['Price_source']].iloc[1:, :]
    df = df.iloc[1:, :]
    df["Predicted_Price"] = np.expm1(model.predict(X_test))

    # Функция для расчёта недель закупки
    def calculate_weeks(row):
        if row["Predicted_Price"] > row["Price"] * 1.02:
            return 6  # Цена растёт (+2%)
        elif row["Predicted_Price"] > row["Price"]:
            return 4  # Незначительный рост
        elif row["Predicted_Price"] < row["Price"] * 0.98:
            return 1  # Цена падает (-2%)
        else:
            return 3  # Стабильная цена → средний объём закупки

    df["Weeks_to_Procure"] = df.apply(calculate_weeks, axis=1)
    return df

# Боковая панель для загрузки файла
with st.sidebar:
    st.header("⚙️ Настройки")
    uploaded_file = st.file_uploader("Загрузите Excel-файл", type=["xlsx"])
    st.markdown("---")
    st.markdown("### Инструкция")
    st.markdown("""
        1. Загрузите файл в формате Excel.
        2. Нажмите кнопку **Выполнить прогнозирование**.
        3. Результаты будут отображены ниже.
    """)

if uploaded_file is not None:
    # Чтение файла
    df = pd.read_excel(uploaded_file)

    # Показываем загруженные данные
    st.subheader("📂 Загруженные данные")
    st.dataframe(df.head(), use_container_width=True)

    # Обработка данных и предсказание
    if st.button("🚀 Выполнить прогнозирование", type="primary"):
        with st.spinner("Выполняется прогнозирование..."):
            result_df = process_data(df)

            # Показываем результаты
            st.subheader("📊 Результаты прогнозирования")
            st.dataframe(result_df, use_container_width=True)

            # Визуализация результатов
            st.subheader("📈 График прогнозируемой цены")
            fig, ax = plt.subplots()
            ax.plot(result_df["dt"], result_df["Price"], label="Фактическая цена")
            ax.plot(result_df["dt"], result_df["Predicted_Price"], label="Прогнозируемая цена", linestyle="--")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Цена")
            ax.legend()
            st.pyplot(fig)

            # Скачивание результата
            st.subheader("📥 Скачать результаты")
            output = io.BytesIO()
            result_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)

            st.download_button(
                label="Скачать как Excel",
                data=output,
                file_name="predicted_procurement.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )