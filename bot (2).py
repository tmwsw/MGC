import telebot
from telebot import types
# ----------------------------------------------------------------------------
# from telebot import util
# db lib
import sqlite3
# random stuff lib
import time
from datetime import datetime
import random

#-----------------------------------------------------------------------------

# !! CONFIG PART !!
token = '7275886313:AAHXR9kY9YFXvd4ewdufHnyFRlswCx-v7wQ'
bot = telebot.TeleBot(token=token)

#------------------------------------------------------------------------------

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 
    "Здравствуйте! Я ваш бот👋\n"
    "Используйте /help, чтобы узнать список команд"
    )

# Обработчик команды /help
@bot.message_handler(commands=['help'])
def help(message):
    bot.reply_to(message,
        "Список доступных команд:\n"
        "/start - Начать работу с ботом\n"
        "/help - Список команд"
    )



bot.polling(none_stop=True)

#работает ураааааа