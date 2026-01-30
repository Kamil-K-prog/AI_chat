# Руководство по форматам ввода и вывода: Google GenAI и OpenAI

Этот документ описывает структуру данных, поддерживаемые медиаформаты и механизмы работы с Chain-of-Thought (мыслями) в библиотеках `google-genai` и `openai`.

---

## 1. Google GenAI (SDK v1+)

Библиотека `google-genai` использует строгую типизацию через модуль `types`. Основная структура взаимодействия — это объекты `Content`, состоящие из списка `Part`.

### Структурирование ввода (Multimodal)

Ввод в `client.models.generate_content` передается через аргумент `contents`, который может быть строкой, словарем или списком объектов `types.Content`.

#### Основные типы Part:

1.  **Text**: `types.Part.from_text(text="...")` — обычный текст.
2.  **Inline Data (Binary)**: `types.Part.from_bytes(data=b"...", mime_type="image/jpeg")` — для небольших файлов.
3.  **File Data (URI)**: `types.Part.from_uri(file_uri="https://...", mime_type="video/mp4")` — ссылка на файл, загруженный через File API или доступный удаленно.

#### Пример мультимодального запроса:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="Что на этом изображении?"),
                types.Part.from_bytes(
                    data=open("image.jpg", "rb").read(),
                    mime_type="image/jpeg"
                )
            ]
        )
    ]
)
```

### Поддерживаемые медиаформаты (Google)

| Категория       | MIME-типы                                                                      | Особенности                          |
| :-------------- | :----------------------------------------------------------------------------- | :----------------------------------- |
| **Изображения** | `image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif`            | До 3072x3072 (сжимаются)             |
| **Аудио**       | `audio/wav`, `audio/mp3`, `audio/aiff`, `audio/aac`, `audio/ogg`, `audio/flac` | 1 сек = ~32 токена                   |
| **Видео**       | `video/mp4`, `video/mpeg`, `video/mov`, `video/avi`, `video/webm` и др.        | 1 сек = ~300 токенов (при 1 FPS)     |
| **Документы**   | `application/pdf`                                                              | До 1000 страниц, 1 стр = 258 токенов |

> **ВАЖНО (URL)**: Google GenAI **НЕ поддерживает** внешние HTTP/HTTPS ссылки напрямую. Вы должны сначала загрузить файл через `client.files.upload()` и использовать полученный `uri`.

---

### Механизм "Мыслей" (Thinking) в Gemini

Модели серии Gemini 2.0+ поддерживают явную выдачу процесса рассуждений.

#### Настройка (Config):

Для включения мыслей необходимо использовать `thinking_config`.

- `include_thoughts: True` — включает выдачу рассуждений.
- `thinking_budget`: Количество токенов, выделяемых на рассуждения (для Gemini 2.x).
- `thinking_level`: `MINIMAL`, `LOW`, `MEDIUM`, `HIGH` (для Gemini 3.x).

#### Обработка ответа:

Ответ приходит в виде списка `parts`. Те части, которые являются рассуждениями, имеют флаг `thought=True`.

```python
config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(include_thoughts=True)
)

response = client.models.generate_content(
    model="gemini-2.0-flash-thinking-exp",
    contents="Реши задачу: верблюд прошел 100 км за 2 дня...",
    config=config
)

for part in response.candidates[0].content.parts:
    if part.thought:
        print(f"РАССУЖДЕНИЕ: {part.text}")
    elif part.text:
        print(f"ОТВЕТ: {part.text}")
```

---

## 2. OpenAI

OpenAI использует более "плоскую" структуру JSON-объектов в массиве `messages`.

### Структурирование ввода

Ввод для мультимодальных моделей (например, `gpt-4o`) передается в поле `content` как список объектов.

#### Пример (Текст + Изображение):

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Опиши картинку"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high" # auto, low, high
                    }
                }
            ]
        }
    ]
)
```

### Поддерживаемые медиаформаты (OpenAI)

| Категория         | Форматы / Методы                                | Особенности                                          |
| :---------------- | :---------------------------------------------- | :--------------------------------------------------- |
| **Изображения**   | `png`, `jpeg`, `webp`, `gif` (не анимированный) | URL или Base64. Лимит 20MB на файл.                  |
| **Аудио (Ввод)**  | `mp3`, `wav`, `flac`, `opus` и др.              | Через API транскрипции (Whisper) или `gpt-4o-audio`. |
| **Аудио (Вывод)** | `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`      | Параметр `audio` в `response_format`.                |

> **ВАЖНО (URL)**: OpenAI **поддерживает** внешние общедоступные ссылки (HTTP/HTTPS) для изображений. Сервер OpenAI самостоятельно скачивает файл для обработки.

---

### Механизм "Мыслей" (Reasoning) в OpenAI (o1, o3)

Модели `o1` и `o3` используют внутренний Chain-of-Thought, который теперь можно частично или полностью извлекать.

#### Настройка:

Используется параметр `reasoning_effort` (заменяет `thinking_budget` в новых версиях).

- Значения: `low`, `medium`, `high`.

#### Извлечение мыслей:

В последних версиях библиотеки мысли возвращаются в поле `reasoning_content` объекта сообщения.

```python
response = client.chat.completions.create(
    model="o1-preview",
    messages=[{"role": "user", "content": "Сложная логическая задача..."}],
    # reasoning_effort="high" # если поддерживается моделью
)

# Извлечение мыслей (если доступно в API)
thought = response.choices[0].message.reasoning_content
print(f"Мысли модели: {thought}")
print(f"Ответ: {response.choices[0].message.content}")
```

**Важно**: Токены рассуждений (reasoning tokens) всегда учитываются в `usage.completion_tokens`, даже если сам текст мыслей скрыт.

---

## Сравнительная таблица Chain-of-Thought

| Характеристика      | Google GenAI (Gemini)                         | OpenAI (o1/o3)                   |
| :------------------ | :-------------------------------------------- | :------------------------------- |
| **Включение**       | `include_thoughts=True` в конфиге             | Автоматически (для o-серии)      |
| **Доступ к тексту** | Поле `part.thought` в списке частей           | Поле `message.reasoning_content` |
| **Контроль усилий** | `thinking_level` (enum)                       | `reasoning_effort` (string)      |
| **Контекст**        | Требует `thought_signature` для Stateless API | Скрыто внутри сессии (Chat)      |

## Рекомендации по использованию типизации (GenAI types)

Всегда импортируйте `types` из `google.genai`. Это обеспечит автодополнение и валидацию:

```python
from google.genai import types

# Правильный способ создания схемы для структурированного вывода
schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "score": types.Schema(type=types.Type.INTEGER),
        "reason": types.Schema(type=types.Type.STRING)
    },
    required=["score", "reason"]
)
```
