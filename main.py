import os
import io
import requests
import numpy as np
from PIL import Image
from aiogram import Bot, Dispatcher, types
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as krs_image

# Устанавливаем токен вашего бота
API_TOKEN = '6455147941:AAHctfTIuZczfPkcdtyW57RwJczTqSnmZEs'

# Путь к вашей обученной модели
MODEL_PATH = 'path/to/your/model.h5'

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Загрузка обученной модели
model = load_model(MODEL_PATH)
model._make_predict_function()  # Обязательно для использования вместе с TensorFlow


# Функция для изменения размера изображения до 250x250 и предсказания
async def process_image(image_bytes):
    try:
        # Чтение изображения из байтов
        img = Image.open(io.BytesIO(image_bytes))

        # Изменение размера изображения до 250x250
        img = img.resize((250, 250))

        # Преобразование изображения в массив numpy
        img_array = krs_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Предсказание с помощью модели
        prediction = model.predict(img_array)

        # Пример вывода предсказания (замените на вашу логику)
        result = f"Предсказание: {prediction}"

        return result
    except Exception as e:
        return f"Ошибка обработки изображения: {e}"


# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply("Привет! Отправь мне изображение для предсказания.")


# Обработчик изображений, полученных от пользователя
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_image(message: types.Message):
    # Получение информации о фото
    photo = message.photo[-1]
    file_id = photo.file_id

    # Запрос на получение файла по его идентификатору
    file = await bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file.file_path}"

    # Скачивание изображения
    image_response = requests.get(file_url)
    image_bytes = image_response.content

    # Обработка изображения и получение предсказания
    result = await process_image(image_bytes)

    # Отправка результата пользователю
    await message.reply(result)


if __name__ == '__main__':
    # Запуск бота
    executor = dp.executor
    executor.start_polling(dp, skip_updates=True)
