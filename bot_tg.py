import os
import io
import logging
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv, dotenv_values
from aiogram import Bot, Dispatcher, types
from aiogram.types import FSInputFile
from aiogram.filters import Command
from aiogram.types import Message
from catboost import CatBoostRegressor

# Загрузка токена из .env
load_dotenv()
TOKEN = os.getenv("MY_KEY")

if not TOKEN:
    raise ValueError("Ошибка: токен бота не найден. Проверьте файл .env")

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Загрузка модели
model = CatBoostRegressor()
model.load_model("catboostmodel_Marin.cbm")  # Укажите путь к вашей модели

def process_data(df):
    df["dt"] = pd.to_datetime(df["dt"], dayfirst=True)
    df.rename(columns={'Цена на арматуру': 'Price'}, inplace=True, errors='ignore')
    
    if 'Price' in df.columns:
        df["Price_source"] = df["Price"].shift(1)
        df["Price_source"] = df["Price_source"].interpolate(method='linear')
        df["Price_source"].fillna(df["Price_source"].mean(), inplace=True)
    
    X_test = df[['Price_source']].iloc[1:, :]
    df = df.iloc[1:, :]
    df["Predicted_Price"] = np.expm1(model.predict(X_test))
    
    def calculate_weeks(row):
        if row["Predicted_Price"] > row["Price"] * 1.02:
            return 6
        elif row["Predicted_Price"] > row["Price"]:
            return 4
        elif row["Predicted_Price"] < row["Price"] * 0.98:
            return 1
        else:
            return 3
    
    df["Weeks_to_Procure"] = df.apply(calculate_weeks, axis=1)
    return df

@dp.message(Command("start"))
async def send_welcome(message: Message):
    await message.answer("Привет! Отправь мне Excel-файл, и я сделаю прогноз закупок.")

@dp.message(lambda message: message.document)
async def handle_document(message: Message):
    file_id = message.document.file_id
    file = await bot.download(file_id)
    df = pd.read_excel(io.BytesIO(file.read()))
    
    result_df = process_data(df)
    
    # Создание графика
    fig, ax = plt.subplots()
    ax.plot(result_df["dt"], result_df["Price"], label="Фактическая цена")
    ax.plot(result_df["dt"], result_df["Predicted_Price"], label="Прогнозируемая цена", linestyle="--")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена")
    ax.legend()
    
    # Сохранение графика
    plot_io = io.BytesIO()
    plt.savefig(plot_io, format='png')
    plot_io.seek(0)
    
    # Подготовка Excel-файла
    output = io.BytesIO()
    result_df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    
    await message.answer_photo(photo=plot_io, caption="📊 График прогнозируемых цен")
    await message.answer_document(FSInputFile(output, filename="predicted_procurement.xlsx"))

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
