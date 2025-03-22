import asyncio
import pandas as pd
from aiogram import Bot, Router, Dispatcher, types, F
from aiogram.types import FSInputFile
from aiogram.filters import Command
from catboost import CatBoostRegressor

TOKEN = "7275886313:AAHXR9kY9YFXvd4ewdufHnyFRlswCx-v7wQ"
bot = Bot(token=TOKEN)
router = Router()
dp = Dispatcher()
dp.include_router(router)

# Загрузка модели
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

@router.message(Command("start", "help"))
async def send_welcome(message: types.Message):
    await message.reply("Отправьте мне файл test.xlsx, и я пришлю прогноз закупки (N недель от 1 до 6).")

@router.message(F.document)
async def handle_file(message: types.Message):
    document_id = message.document.file_id
    file_info = await bot.get_file(document_id)
    file_path = file_info.file_path
    downloaded_file = await bot.download_file(file_path)
    
    # Сохранение файла
    with open("test.xlsx", "wb") as new_file:
        new_file.write(downloaded_file.read())
    
    # Чтение файла и подготовка данных
    test_df = pd.read_excel("test.xlsx")
    test_df["dt"] = pd.to_datetime(test_df["dt"], dayfirst=True)
    test_df["year"] = test_df["dt"].dt.year
    test_df["month"] = test_df["dt"].dt.month
    test_df["day"] = test_df["dt"].dt.day
    test_df["week"] = test_df["dt"].dt.isocalendar().week
    
    # Переименование столбца Price в Price_source, если это необходимо
    test_df.rename(columns={'Price': 'Price_source'}, inplace=True)
    
    # Генерация лагов для Price_source
    for i in range(1, 7):
        test_df[f"lag_{i}"] = test_df["Price_source"].shift(i)
    
    # Проверка наличия всех необходимых признаков
    feature_columns = model.feature_names_
    # Если все еще есть отсутствующие признаки, добавьте их здесь
    
    # Удаление строк с пропусками
    test_df = test_df.dropna()
    X_test = test_df[feature_columns]
    
    # Предсказание
    test_df["Прогноз_цены"] = model.predict(X_test)
    
    # Определение закупки (используем Price_source вместо Price)
    def calculate_weeks(row):
        if row["Прогноз_цены"] > row["Price_source"] * 1.02:
            return 6
        elif row["Прогноз_цены"] > row["Price_source"]:
            return 4
        elif row["Прогноз_цены"] < row["Price_source"] * 0.98:
            return 1
        else:
            return 3
    
    test_df["Закупка_недель"] = test_df.apply(calculate_weeks, axis=1)
    
    # Отправка результата
    await message.reply(f"Рекомендуемая закупка: {test_df['Закупка_недель'].iloc[-1]} недель")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())