# Руководство по форматам: Google GenAI и OpenAI

Справочник для написания конвертеров УФС ↔ нативные форматы.

---

## 1. Роли сообщений


| УФС      | Google GenAI                                               | OpenAI                |
| ----------- | ---------------------------------------------------------- | --------------------- |
| `system`    | `system_instruction` (отдельный параметр) | `"role": "system"`    |
| `user`      | `"role": "user"`                                           | `"role": "user"`      |
| `assistant` | `"role": "model"`                                          | `"role": "assistant"` |
| `tool`      | `"role": "user"` с `function_response` Part               | `"role": "tool"`      |

> **GenAI**: Одинаковые роли НЕ могут идти подряд! При конвертации нужно объединять.

---

## 2. Структура Content

### Google GenAI

```python
types.Content(
    role="user" | "model",
    parts=[
        types.Part.from_text(text="..."),
        types.Part.from_bytes(data=b"...", mime_type="image/jpeg"),
        types.Part.from_uri(file_uri="https://...", mime_type="video/mp4"),
        # function_call и function_response — автоматически
    ]
)
```

### OpenAI

```python
{
    "role": "user" | "assistant" | "system" | "tool",
    "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "auto"}}
    ],
    # Для assistant с tool_calls:
    "tool_calls": [...],
    # Для tool:
    "tool_call_id": "call_xxx"
}
```

---

## 3. Медиаформаты

### Google GenAI


| Категория     | MIME-типы                                    | Примечания           |
| ---------------------- | ------------------------------------------------ | ------------------------------ |
| Изображения | `image/png`, `jpeg`, `webp`, `heic`, `heif`      | До 3072x3072                 |
| Аудио             | `audio/wav`, `mp3`, `aiff`, `aac`, `ogg`, `flac` | 1 сек ≈ 32 токена    |
| Видео             | `video/mp4`, `mpeg`, `mov`, `avi`, `webm`        | 1 сек ≈ 300 токенов |
| Документы     | `application/pdf`                                | До 1000 стр               |

> **URL**: НЕ поддерживает внешние HTTP ссылки. Только `client.files.upload()` → `uri`.

### OpenAI


| Категория     | Форматы               | Примечания                    |
| ---------------------- | ---------------------------- | --------------------------------------- |
| Изображения | `png`, `jpeg`, `webp`, `gif` | URL или Base64. До 20MB            |
| Аудио (ввод)  | `mp3`, `wav`, `flac`, `opus` | Через Whisper или`gpt-4o-audio` |

> **URL**: Поддерживает внешние HTTP/HTTPS ссылки для изображений.

---

## 4. Function Calling (Tools)

### 4.1 Объявление функций

**Google GenAI:**

```python
config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[
        {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    ])]
)
```

**OpenAI:**

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]
```

### 4.2 Получение tool_call из ответа

**Google GenAI:**

```python
for part in response.candidates[0].content.parts:
    if part.function_call:
        name = part.function_call.name
        args = part.function_call.args  # dict
        # id отсутствует в GenAI!
```

**OpenAI:**

```python
for tool_call in response.choices[0].message.tool_calls:
    id = tool_call.id           # "call_xxx" — обязательно сохранить!
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)  # строка → dict
```

### 4.3 Отправка tool_result

**Google GenAI (роль `"user"`):**

```python
types.Content(
    role="user",
    parts=[
        types.Part.from_function_response(
            name="get_weather",
            response={"result": {"temp": 20, "unit": "C"}}
        )
    ]
)
```

**OpenAI (роль `"tool"`):**

```python
{
    "role": "tool",
    "tool_call_id": "call_xxx",  # Должен совпадать с id из tool_call!
    "content": '{"temp": 20, "unit": "C"}'  # Строка JSON
}
```

### 4.4 Мультимодальный tool_result (Gemini 3+)

```python
types.Content(
    role="user",  # ВАЖНО: role="user", НЕ "tool"!
    parts=[
        types.Part.from_function_response(
            name="generate_chart",
            response={"image_ref": {"$ref": "chart.png"}},
            parts=[
                types.FunctionResponsePart(
                    inline_data=types.FunctionResponseBlob(
                        mime_type="image/png",
                        display_name="chart.png",
                        data=image_bytes
                    )
                )
            ]
        )
    ]
)
```

> OpenAI **не поддерживает** мультимодальные tool_result.

---

## 5. Структура Response

### Google GenAI

```python
response.candidates[0].content.parts[]  # List[Part]
# Каждый Part может быть:
#   - part.text (строка)
#   - part.thought (bool) + part.text — мысли модели
#   - part.function_call — вызов инструмента

response.candidates[0].finish_reason  # "STOP", "TOOL_USE", "MAX_TOKENS"
response.usage_metadata.prompt_token_count
response.usage_metadata.candidates_token_count
```

### OpenAI

```python
response.choices[0].message.content      # Текст ответа
response.choices[0].message.tool_calls   # Список вызовов [ChatCompletionMessageToolCall]
response.choices[0].message.reasoning_content  # Мысли (o1/o3)

response.choices[0].finish_reason  # "stop", "tool_calls", "length", "content_filter"
response.usage.prompt_tokens
response.usage.completion_tokens
```

### Маппинг finish_reason


| УФС           | GenAI        | OpenAI           |
| ---------------- | ------------ | ---------------- |
| `stop`           | `STOP`       | `stop`           |
| `tool_calls`     | `TOOL_USE`   | `tool_calls`     |
| `length`         | `MAX_TOKENS` | `length`         |
| `content_filter` | `SAFETY`     | `content_filter` |

---

## 6. Thinking (Chain-of-Thought)

### Google GenAI

```python
config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(include_thoughts=True)
)

for part in response.candidates[0].content.parts:
    if part.thought:
        print(f"МЫСЛЬ: {part.text}")
    else:
        print(f"ОТВЕТ: {part.text}")
```

### OpenAI (o1/o3)

```python
thought = response.choices[0].message.reasoning_content  # Может быть None
answer = response.choices[0].message.content
```

> **УФС**: Мысли сохраняются как `ThoughtContent(type="thought", text="...")`.
> При конвертации в `interleaved` режиме — фильтруются перед финальным ответом.

---

## 7. Ключевые ограничения


| Ограничение                                         | GenAI                                  | OpenAI                          |
| -------------------------------------------------------------- | -------------------------------------- | ------------------------------- |
| Последовательные одинаковые роли | ❌ Запрещено                  | ✅ Разрешено           |
| Внешние URL для медиа                           | ❌ Только через Files API   | ✅ Поддерживаются |
| ID у tool_call                                                | ❌ Отсутствует              | ✅ Обязателен         |
| Мультимодальный tool_result                     | ✅ Gemini 3+                           | ❌ Нет                       |
| system role                                                    | Отдельный`system_instruction` | В массиве`messages`     |
