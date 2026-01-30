Пример структуры УФС:

```python
messages_structure = {
    "chat_metadata": {  # Метаданные и настройки всего чата
        "version": "1.0",  # Версия формата УФС (не изменяется)
        "config": {
            "thinking_mode": "interleaved" | "preserved",
            "provider": "openai" | "genai",
            "...": "..."
        },
        # Кэш контента для GenAI (опционально)
        "cache_ref": {
            "genai": {
                "name": "cachedContents/abc123",
                "expires_at": "ISO-TIMESTAMP"
            }
        }
    },
    "messages": [
        {
            "id": "msg_uuid123",  # Уникальный ID сообщения
            "timestamp": "2026-01-30T13:45:00+05:00",  # ISO 8601

            # Роль сообщения. Одна и та же роль не может идти подряд (требование Gemini).
            # Если модель возвращает отдельно thought, затем text, затем tool_call — собирается в один блок.
            "role": "system" | "user" | "assistant" | "tool",

            # Имя участника (опционально, для multi-user сценариев)
            "name": "Камиль",

            "content": [
                {
                    # Тип контента
                    "type": "text" | "thought" | "image" | "audio" | "video" | "document" | "tool_call" | "tool_result",

                    # ─── Для текста и мыслей (type: "text" | "thought") ───
                    "text": "Строка контента",

                    # ─── Для медиа-файлов и документов (type: "image" | "audio" | "video" | "document") ───
                    # Также используется в tool_result, если инструмент возвращает файлы
                    "assets": [  # Массив, т.к. может быть несколько файлов
                        {
                            "id": "file123",  # Внутренний ID
                            "local_path": "files/image/pic.jpg",
                            "mime_type": "image/jpeg",  # Или "application/pdf", "text/plain" для документов
                            "size_bytes": 102400,
                            "data_base64": "...",  # Опционально для маленьких файлов
                            "cloud_refs": {
                                "openai": {"id": "file-123"},
                                "genai": {
                                    "uri": "https://generativelanguage.googleapis.com/...",
                                    "expires_at": "ISO-TIMESTAMP"  # Файлы в Gemini живут 48 часов
                                }
                            }
                        }
                    ],

                    # ─── Для вызовов инструментов (type: "tool_call") ───
                    "tool_call": {
                        "id": "call_abc",
                        "name": "func_name",
                        "args": {}
                    },

                    # ─── Для ответов инструментов (type: "tool_result") ───
                    "tool_result": {
                        "id": "call_abc",  # Должен совпадать с id из tool_call
                        "name": "func_name",
                        "content": "текстовый результат",
                        "is_error": False
                    }
                    # Если инструмент возвращает файлы, добавляется "assets": [...]
                }
            ],

            # ─── Метаданные генерации (только для role: "assistant") ───
            "metadata": {
                "model": "gpt-4o",
                "model_class": "openai" | "genai",  # Какой тип модели использовался
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "total_tokens": 300
                },
                "finish_reason": "stop" | "tool_calls" | "length" | "content_filter",
                "latency_ms": 1234  # Время генерации в миллисекундах
            }
        }
    ]
}
```
