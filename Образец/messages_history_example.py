"""
Пример структуры массива-адаптера сообщений. Два пути развития:
    1) Модели внутри себя ведут параллельную цепочку сообщений, которая не пересекается с той, что хранится в  модуле UI
    2) На вход модели передаётся такой формат истории. Перед каждой генерацией он конвертируется в genai- или openai-совместимый, после генерации конвертируется обратно и возвращается из функции generate()   // Предпочтительно. Главное, чтобы циклы конвертации туда-обратно не занимали много времени

Текущая структура (массив словарей с 'role' и 'content'):
    * role: "system|user|model|tool_responses"
    * content: список элементов [{"type": "...", "data": ...}], где:
        - text: {"type": "text", "data": str} — текстовое сообщение.
        - thoughts: {"type": "thoughts", "data": str} — мысли модели (только для роли model), опционально.
        - media: {"type": "media", "data": [{"type": "image|audio|video", "source_type": "base64|url|path", "content": str}, ...]} — медиа-контент.
        - tool_calls: {"type": "tool_calls", "data": [{"name": str, "id": str, "args": dict}, ...]} — вызовы инструментов (model). `id` используется при конвертации в openai-совместимый JSON
        - tool_response: {"type": "tool_response", "data": {"name": str, "id": str, "error": bool, "content": str}} — результат выполнения инструмента (tool_responses).
    * metadata: данные о генерации (только у сообщений с ролью model). Пока точно не ясно, что конкретно будет хранится. Можно будет выводить в UI по нажатию на кнопку
        - usage: {"input": int, "output": int} - количество токенов на входе и выходе
        - model_name: str - название модели, сгенерировавшей ответ
"""

example_image_path = "example.jpg"
example_audio_path = "example.mp3"

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "data": "Ты - полезный ИИ агент"},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "data": "Привет."},
        ],
    },
    {
        "role": "model",
        "content": [
            {
                "type": "thoughts",
                "data": "Пользователь написал 'Привет'. Отвечу и спрошу, как я могу помочь.",
            },
            {"type": "text", "data": "Привет! Как я могу помочь?"},
        ],
        "metadata": {
            "usage": {
                "input": 10,
                "output": 10,
            },
            "model_name": "gpt-4o",
        }
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "data": "Просмотри эту картинку. Что ты видишь?"},
            {
                "type": "media",
                "data": [
                    {
                        "type": "image",  # Доступные типы: audio, video, image
                        "source_type": "path",
                        "content": example_image_path,
                    },
                    {
                        "type": "audio",
                        "source_type": "path",
                        "content": example_audio_path,
                    },
                ],
            },
        ],
    },
    {
        "role": "model",
        "content": [
            {"type": "thoughts","data": "Пользователь показал мне картинку. Отвечу на его вопрос."},
            {"type": "text", "data": "Это красивый лошадиный двор."},
        ],
        "metadata": {
            "usage": {
                "input": 200,
                "output": 100,
            },
            "model_name": "gpt-4o",
        }
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "data": "Какая погода сейчас? Для погоды в Москве предоставь корень из числа, для Оренбурга - само значение."},
        ],
    },
    {
        "role": "model",
        "content": [
            {"type": "thoughts", "data": "Пользователь спросил, какая погода сейчас. Мне нужно вызвать инструмент get_weather() для Москвы и Оренбурга, затем sqr() для Москвы и предоставить ответ."},
            {"type": "tool_calls", "data": [
                {
                    "tool_name": "get_weather",
                    "id": "1",
                    "args": {"city": "Moscow"},
                },
                {
                    "tool_name": "get_weather",
                    "id": "2",
                    "args": {"city": "Orenburg"},
                },
            ]},
        ],
        "metadata": {
            "usage": {
                "input": 200,
                "output": 100,
            },
            "model_name": "gpt-4o",
        }
    },
    {
        "role": "tools_responses",
        "content": [
            {"type": "tool_response", "data": {"tool_name": "get_weather", "id": "1", "error": False, "response": "25 C"}},
            {"type": "tool_response", "data": {"tool_name": "get_weather", "id": "2", "error": False, "response": "18 C"}},
        ],
    },
    {
        "role": "model",
        "content": [
            {"type": "thoughts", "data": "Так, температура в Москве сейчас 25 градусов по Цельсию, а в Оренбурге 18. Надо вычислить корень из 25 (погода в Москве)"},
            {"type": "tool_calls", "data": [{"tool_name": "sqr", "id": "3", "args": {"number": "25"}}]},
        ],
        "metadata": {
            "usage": {
                "input": 200,
                "output": 100,
            },
            "model_name": "gpt-4o",
        }
    },
    {
        "role": "tools_responses",
        "content": [
            {"type": "tool_response", "data": {"tool_name": "sqr", "id": "3", "error": False, "response": "5"}},
        ],
    },
    {
        "role": "model",
        "content": [
            {"type": "thoughts", "data": "Корень дал число 5. Это и есть ответ. Предоставлю его пользователю."},
            {"type": "text", "data": "Сейчас в Москве 25 градусов по Цельсию. Корень из этого числа равен числу 5. Погода в Оренбурге 18 градусов по Цельсию"},
        ],
        "metadata": {
            "usage": {
                "input": 200,
                "output": 100,
            },
            "model_name": "gpt-4o",
        }
    },
    {"role": "user", "content": {"text": "Нарисуй погоду картинкой"}},
    {
        "role": "model",
        "content": [
            {"type": "thoughts", "data": "Пользователь хочет увидеть картинку погоды. До этого я узнал погоду в Москве и Оренбруге, поэтому надо включить оба прогноза в итоговую картинку. Начну рисовать."},
            {"type": "text", "data": "Конечно! Вот карточка, отображающая текущую погоду в Москве и Оренбурге:"},
            {"type": "media", "data": [
                {
                    "type": "image",
                    "source_type": "url",
                    "content": "https://example.com/image.jpg",
                }
            ]},
        ],
        "metadata": {
            "usage": {
                "input": 200,
                "output": 100,
            },
            "model_name": "gpt-4o",
        }
    },
]
