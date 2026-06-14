# Работа с файлами через Google GenAI Files API

Google GenAI SDK (библиотека `google-genai`) предоставляет специальный Files API для работы с большими медиафайлами (такими как длинные видео, аудиозаписи, объемные PDF-документы и изображения). Использование Files API позволяет передавать файлы модели по ссылке (URI) вместо их встраивания непосредственно в запрос в кодировке base64.

---

## ⌛ Время хранения файлов (Retention Policy)

> [!IMPORTANT]
> **Срок жизни файлов: 48 часов**
> Все файлы, загруженные через Files API, хранятся в облаке Google **не более 48 часов**. По истечении этого времени они автоматически безвозвратно удаляются.

- **Время жизни фиксировано**: Использование файла в запросах генерации контента, чтение его метаданных или обращение к нему через API **НЕ сбрасывает** и **НЕ продлевает** 48-часовой таймер. Механизм «скользящего окна» (sliding window) отсутствует. Файл будет безвозвратно удален ровно через 48 часов после момента его загрузки.
- **Связь с историей чата**: Если ваше приложение сохраняет историю чата локально и планирует повторно использовать старые сообщения с медиафайлами спустя 48 часов, необходимо предусмотреть механизм повторной загрузки файлов перед отправкой запроса, так как старые URI станут недействительными.
- **Безопасность**: Загруженные файлы привязаны к вашему проекту API и доступны для обработки только по вашему API-ключу.

---

## 🚫 Лимиты и ограничения

1. **Максимальный размер одного файла**: **2 ГБ**.
2. **Общий лимит хранилища на проект**: **20 ГБ**. При превышении этого лимита загрузка новых файлов завершится ошибкой. Старые файлы необходимо удалять вручную или дожидаться их автоматического удаления.
3. **Ограничение на скачивание**: Загруженные файлы **нельзя скачать обратно** из облака Files API. API возвращает только метаданные файла (размер, имя, URI, статус). Само облако используется исключительно как временный буфер для инференса моделей Gemini.
4. **Сравнение с Inline-передачей (передача напрямую в prompt)**:
   - **Inline (base64)**: Подходит для файлов **меньше 20 МБ** (суммарно до 100 МБ на запрос в некоторых моделях). Данные передаются непосредственно в теле запроса и не сохраняются в облаке Files API.
   - **Files API**: Рекомендуется для любых файлов **крупнее 20 МБ** (обязательно для видео и больших документов).

---

## 📂 Поддерживаемые MIME-типы и форматы

Перед загрузкой файла крайне важно правильно определить его MIME-тип. Модели Gemini чувствительны к некорректным MIME-типам.

### Документы
| Формат | MIME-тип | Примечание |
| :--- | :--- | :--- |
| **PDF** | `application/pdf` | Максимальный размер для PDF часто составляет 50 МБ для некоторых типов интеграций. |
| **Plain Text** | `text/plain` | Обычные текстовые файлы. |

### Изображения
| Формат | MIME-тип |
| :--- | :--- |
| **PNG** | `image/png` |
| **JPEG** | `image/jpeg` |
| **WebP** | `image/webp` |

### Аудио
| Формат | MIME-тип |
| :--- | :--- |
| **WAV** | `audio/wav` |
| **MP3** | `audio/mp3` |
| **AAC** | `audio/aac` |
| **OGG** | `audio/ogg` |
| **FLAC** | `audio/flac` |

### Видео
| Формат | MIME-тип |
| :--- | :--- |
| **MP4** | `video/mp4` |
| **MPEG** | `video/mpeg` |
| **MOV** (QuickTime) | `video/quicktime` |
| **AVI** | `video/avi` |
| **WebM** | `video/webm` |
| **WMV** | `video/wmv` |
| **3GPP** | `video/3gpp` |

---

## 💻 Примеры использования на Python (SDK `google-genai`)

Убедитесь, что у вас установлена актуальная версия библиотеки:
```bash
pip install -U google-genai
```

### 1. Загрузка файла и ожидание обработки

При загрузке больших файлов (особенно видео) Google запускает асинхронную обработку (транскодирование, извлечение признаков). Файл **нельзя** передавать модели, пока его статус не сменится на `ACTIVE`.

```python
import time
from google import genai
from google.genai import errors

# Инициализируем клиент (API-ключ берется из переменной окружения GEMINI_API_KEY)
client = genai.Client()

print("Загрузка видеофайла...")
video_file = client.files.upload(file="sample_video.mp4")

# Имя файла в системе Google (имеет формат "files/xxxxxxxxxxxx")
file_name = video_file.name
print(f"Файл загружен с именем: {file_name}")

# Опрашиваем API, пока статус обработки не изменится
while video_file.state.name == "PROCESSING":
    print("Файл обрабатывается в облаке, ожидание 10 секунд...")
    time.sleep(10)
    # Обновляем метаданные файла
    video_file = client.files.get(name=file_name)

if video_file.state.name == "FAILED":
    raise RuntimeError(f"Обработка файла не удалась: {video_file.error.message}")

print("Файл готов к использованию!")
```

### 2. Отправка запроса генерации с файлом

После того как файл перешел в статус `ACTIVE`, его можно передавать в параметр `contents` метода `generate_content`.

```python
# Отправка запроса с использованием загруженного файла
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "Подробно опиши, что происходит на этом видео с указанием таймкодов.",
        video_file
    ]
)

print("\nОтвет модели:")
print(response.text)
```

### 3. Проверка доступности и срока хранения файла по его имени (ID)

Если у вас сохранен только строковый идентификатор файла (поле `name` вида `files/xxxxxxxxxxxx`), вы можете в любой момент получить его актуальные метаданные из облака с помощью `client.files.get()`. Это позволяет узнать, завершилась ли его обработка и сколько времени осталось до его автоматического удаления.

```python
from datetime import datetime, timezone

# Предположим, у нас есть сохраненный ID файла
file_name = "files/your-unique-file-id"

try:
    # Запрашиваем информацию о файле по его имени
    file_info = client.files.get(name=file_name)
    
    # 1. Проверяем статус доступности
    if file_info.state.name == "ACTIVE":
        print("Файл доступен и готов к использованию!")
    elif file_info.state.name == "PROCESSING":
        print("Файл все еще обрабатывается.")
    elif file_info.state.name == "FAILED":
        print("Обработка файла завершилась неудачей.")
        
    # 2. Вычисляем, сколько времени осталось до удаления (из лимита 48 часов)
    # expiration_time представляет собой datetime объект в UTC
    expires_at = file_info.expiration_time
    if expires_at:
        now = datetime.now(timezone.utc)
        remaining = expires_at - now
        if remaining.total_seconds() > 0:
            print(f"Файл будет храниться еще: {remaining}")
            print(f"Время удаления (UTC): {expires_at}")
        else:
            print("Срок хранения файла истек.")
            
except errors.APIError as e:
    # Если файл уже удален (истекли 48 часов или удален вручную), API вернет ошибку 404
    if e.code == 404:
        print("Файл не найден в облаке. Возможно, он был удален или истекли 48 часов.")
    else:
        print(f"Ошибка API при получении данных файла: {e}")
```

### 4. Просмотр списка загруженных файлов

Вы можете получить список всех файлов, которые в данный момент хранятся в вашем облаке (не истекли 48 часов и не были удалены).

```python
print("Список загруженных файлов:")
for f in client.files.list():
    print(f"- Имя: {f.name} | Отображаемое имя: {f.display_name} | Статус: {f.state.name} | Создан: {f.created_time}")
```

### 5. Ручное удаление файлов

Чтобы не упираться в лимит 20 ГБ, рекомендуется явно удалять файлы после того, как они больше не нужны для работы.

```python
# Удаление файла по его имени (ID)
client.files.delete(name=file_name)
print(f"Файл {file_name} успешно удален из облака.")
```

### 6. Асинхронное использование (AsyncClient)

Если ваше приложение построено на `asyncio`, используйте `genai.Client` аналогичным образом, но с асинхронными вызовами:

```python
import asyncio
from google import genai

async def main():
    client = genai.Client()
    
    # Загрузка
    video_file = await client.aio.files.upload(file="sample_video.mp4")
    
    # Ожидание обработки
    while video_file.state.name == "PROCESSING":
        await asyncio.sleep(10)
        video_file = await client.aio.files.get(name=video_file.name)
        
    if video_file.state.name == "ACTIVE":
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Что изображено?", video_file]
        )
        print(response.text)
        
        # Удаление
        await client.aio.files.delete(name=video_file.name)

asyncio.run(main())
```

### 7. Передача файлов в виде Base64 (Inline-данные)

Для передачи небольших файлов (до 20 МБ) нет необходимости загружать их в облако Files API. Их можно передавать непосредственно в prompt как inline-данные. 

Для этого в библиотеке `google-genai` используется метод `types.Part.from_bytes()`. Он принимает сырые байты (не саму Base64-строку), поэтому предварительно строку Base64 нужно декодировать:

```python
import base64
from google import genai
from google.genai import types

client = genai.Client()

# Пример Base64-строки изображения (например, 1x1 белый пиксель)
base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# 1. Декодируем строку в байты
raw_bytes = base64.b64decode(base64_data)

# 2. Создаем объект Part.from_bytes с указанием MIME-типа
media_part = types.Part.from_bytes(
    data=raw_bytes,
    mime_type="image/png"
)

# 3. Передаем объект в запрос модели
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "Что изображено на этой картинке?",
        media_part
    ]
)

print(response.text)
```

