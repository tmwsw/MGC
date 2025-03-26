import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import random

# Настройка страницы
st.set_page_config(page_title="Рекомендации по закупкам", page_icon="📊", layout="wide")

# Заголовок приложения
st.title("📊 Рекомендации по закупкам")
st.write("Это приложение предоставляет рекомендации по количеству недель для закупки на основе выбранной даты.")

# Загрузка модели и данных
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("main_model.cbm")
    return model

@st.cache_data
def load_historical_data():
    df = pd.read_excel("combined_df.xlsx")
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(by="dt")
    
    # Добавляем дополнительные признаки
    df['price_1week_ago'] = df['Price'].shift(7)
    df['price_1month_ago'] = df['Price'].shift(30)
    df['rolling_7d'] = df['Price'].rolling(7).mean()
    df['rolling_30d'] = df['Price'].rolling(30).mean()
    df['month'] = df['dt'].dt.month
    df['day_of_week'] = df['dt'].dt.dayofweek
    
    df = df.dropna()
    return df

model = load_model()
historical_prices = load_historical_data()

# Улучшенная функция для расчета недель закупки
def calculate_weeks(price, predicted_prices):
    # Рассчитываем изменения цен
    changes = [(p - price)/price*100 for p in predicted_prices]
    
    # Ключевые метрики
    avg_change = np.mean(changes)
    max_change = np.max(changes)
    min_change = np.min(changes)
    volatility = max_change - min_change  # Размах колебаний
    
    # Определяем силу тренда
    trend_strength = abs(avg_change)
    
    # Новая логика с использованием всего диапазона 1-6 недель
    if avg_change > 0:  # При росте цен
        if trend_strength > 5 or volatility > 8:
            return 1, f"🚨 Срочная минимальная закупка (резкий рост до +{max_change:.1f}%)"
        elif trend_strength > 3:
            return 2, f"⚠️ Уменьшенная закупка (сильный рост +{avg_change:.1f}%)"
        elif trend_strength > 1:
            return 3, f"Стандартная закупка (рост +{avg_change:.1f}%)"
        else:
            return 4, f"Нормальная закупка (незначительный рост +{avg_change:.1f}%)"
    else:  # При падении цен
        if trend_strength > 5 or volatility > 8:
            return 6, f"💰 Максимальная закупка (резкое падение {min_change:.1f}%)"
        elif trend_strength > 3:
            return 5, f"📈 Увеличенная закупка (падение {avg_change:.1f}%)"
        elif trend_strength > 1:
            return 4, f"Нормальная закупка (незначительное падение {avg_change:.1f}%)"
        else:
            return 3, f"Стандартная закупка (стабильные цены)"

# Расширенная функция для создания признаков
def create_features(current_price, days_ahead, date):
    features = {
        'Price_source': current_price,
        'days_ahead': days_ahead,
        'day_of_week': date.weekday(),
        'day_of_month': date.day,
        'month': date.month,
        'is_month_end': date.day > 25,
        'price_1week_ago': historical_prices["Price"].iloc[-7],
        'price_1month_ago': historical_prices["Price"].iloc[-30],
        'rolling_7d_avg': historical_prices["rolling_7d"].iloc[-1],
        'rolling_30d_avg': historical_prices["rolling_30d"].iloc[-1],
        'price_change_7d': (current_price - historical_prices["Price"].iloc[-7])/historical_prices["Price"].iloc[-7]*100,
    }
    
    # Сезонные компоненты
    features['sin_month'] = np.sin(2 * np.pi * date.month / 12)
    features['cos_month'] = np.cos(2 * np.pi * date.month / 12)
    
    return features

# Создаем две колонки
col1, col2 = st.columns([1, 3])

# Левая колонка - выбор параметров
with col1:
    st.header("📅 Параметры")
    date_input = st.date_input("Выберите дату", 
                             min_value=datetime.today(), 
                             value=datetime.today(),
                             format="YYYY/MM/DD")
    
    if st.button("Получить рекомендацию", type="primary", use_container_width=True):
        st.session_state['calculate'] = True
    else:
        st.session_state['calculate'] = False

# Правая колонка - результаты и график
with col2:
    if 'calculate' in st.session_state and st.session_state['calculate']:
        current_price = historical_prices["Price"].iloc[-1]
        days_ahead = (date_input - datetime.today().date()).days
        
        # Генерация прогноза на 6 недель вперед от выбранной даты
        future_dates = [date_input + timedelta(weeks=i) for i in range(0, 6)]  # От выбранной даты до +5 недель (всего 6 точек)
        future_prices = []
        
        base_noise = current_price * 0.005  # Фиксированный уровень шума

        for i, date in enumerate(future_dates):
            days = (date - datetime.today().date()).days
            features = create_features(current_price, days, date)
            features_df = pd.DataFrame([features])[model.feature_names_]
            
            predicted = np.expm1(model.predict(features_df)[0])
            
            # Добавляем естественную изменчивость
            noise = np.random.uniform(-base_noise, base_noise) * (i+1)/2
            seasonal_factor = 1 + (features['sin_month'] * 0.01)
            
            final_price = predicted * seasonal_factor + noise
            future_prices.append(final_price)
        
        # Прогноз на выбранную дату - первый элемент
        predicted_price = future_prices[0]
        weeks, recommendation = calculate_weeks(current_price, future_prices)
        
        # График прогноза
        st.header(f"📈 Прогноз с {date_input.strftime('%d.%m.%Y')} на 6 недель")
        
        forecast_df = pd.DataFrame({
            "Дата": future_dates,
            "Прогнозируемая цена": future_prices,
            "Изменение, %": [(price - current_price)/current_price*100 for price in future_prices]
        })
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(forecast_df["Дата"], forecast_df["Прогнозируемая цена"], marker='o', color='#1f77b4', linewidth=2)
        ax.axhline(current_price, color='red', linestyle='--', linewidth=1)

        ax.set_title(f"Прогноз цены с {date_input.strftime('%d.%m.%Y')}")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Цена")
        ax.set_xticks(future_dates)
        ax.set_xticklabels([date.strftime('%d.%m') for date in future_dates])
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Добавляем блок с рекомендацией под графиком
        st.subheader("📌 Рекомендация по закупкам")
        col_rec1, col_rec2 = st.columns([1, 3])
        with col_rec1:
            st.metric("Недель для закупки", weeks)
        with col_rec2:
            st.write(recommendation)
        
        # Таблица с прогнозами
        st.dataframe(forecast_df.style.format({
            "Прогнозируемая цена": "{:,.2f}",
            "Изменение, %": "{:.1f}%"
        }).background_gradient(cmap='Blues', subset=["Изменение, %"]), 
        use_container_width=True)
        
        # Секция сохранения результатов
        st.header("💾 Сохранение результатов")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame({
                "Дата": [date_input.strftime('%d.%m.%Y')],
                "Прогнозируемая цена": [predicted_price],
                "Изменение цены, %": [(predicted_price - current_price)/current_price*100],
                "Рекомендуемое количество недель": [weeks],
                "Рекомендация": [recommendation]
            }).to_excel(writer, sheet_name="Рекомендация", index=False)
            
            forecast_df.to_excel(writer, sheet_name="Прогноз на 6 недель", index=False)
        
        st.download_button(
            label="Скачать полный отчет в Excel",
            data=output.getvalue(),
            file_name=f"рекомендация_закупки_{date_input.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("Выберите дату и нажмите кнопку 'Получить рекомендацию' для отображения результатов")

# Упрощенные пояснения
st.markdown("---")
st.subheader("ℹ️ О системе рекомендаций")
st.write("""
Приложение анализирует исторические данные о ценах и прогнозирует их изменение в будущем. 
На основе этого анализа формируются рекомендации по оптимальному объему закупок.
""")