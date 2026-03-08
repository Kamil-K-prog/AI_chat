"""
Универсальный Формат Сообщений (УФС/UMF) v1.0

Pydantic-модели для хранения истории чата в унифицированном формате.
Данные из УФС конвертируются в формат, совместимый с OpenAI или Google GenAI.
"""

from typing import Optional, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# ОБЛАЧНЫЕ ССЫЛКИ (для загруженных файлов)
# ══════════════════════════════════════════════════════════════════════════════


class CloudRef(BaseModel):
    """
    Универсальная ссылка на файл в облачном хранилище провайдера.
    Может содержать id, uri, expires_at и другие поля.
    """

    id: Optional[str] = None  # file-xxx (OpenAI)
    uri: Optional[str] = None  # https://... (GenAI)
    expires_at: Optional[datetime] = None  # Время истечения срока жизни (GenAI)
    # Разрешаем любые дополнительные поля для специфичных провайдеров
    model_config = {"extra": "allow"}


class CloudRefs(BaseModel):
    """
    Ссылки на файл в облачных хранилищах различных провайдеров.
    Поля openai и genai типизированы для удобства, остальные можно передавать как kwargs.
    """

    public: Optional[CloudRef] = None  # Публичная ссылка (исходник в вебе)
    openai: Optional[CloudRef] = None
    genai: Optional[CloudRef] = None
    # Разрешаем добавлять других провайдеров (deepseek, anthropic и т.д.)
    model_config = {"extra": "allow"}


# ══════════════════════════════════════════════════════════════════════════════
# ASSET (универсальная модель для файлов)
# ══════════════════════════════════════════════════════════════════════════════


class Asset(BaseModel):
    """
    Универсальная модель для медиа-файлов и документов.

    Используется для:
    - Изображений, аудио, видео, документов в сообщениях
    - Файлов, возвращаемых инструментами
    """

    id: str  # Внутренний уникальный ID
    type: Literal["image", "audio", "video", "document"]  # Тип ассета
    local_path: Optional[str] = None  # Путь к файлу на диске (если есть)
    mime_type: str  # image/jpeg, audio/mp3, application/pdf и т.д.
    size_bytes: int = 0
    data_base64: Optional[str] = None  # Для небольших файлов (inline)
    cloud_refs: Optional[CloudRefs] = None  # Ссылки на загруженные копии


# ══════════════════════════════════════════════════════════════════════════════
# TOOL CALL / TOOL RESULT
# ══════════════════════════════════════════════════════════════════════════════


class ToolCall(BaseModel):
    """Вызов инструмента моделью."""

    id: str  # call_xxx
    name: str  # Имя функции
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Результат выполнения инструмента."""

    id: str  # Должен совпадать с id из ToolCall
    name: str  # Имя функции
    content: str  # Текстовый результат
    is_error: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT ITEMS (элементы содержимого сообщения)
# ══════════════════════════════════════════════════════════════════════════════


class TextContent(BaseModel):
    """Текстовое содержимое."""

    type: Literal["text"] = "text"
    text: str


class ThoughtContent(BaseModel):
    """Мысли модели (reasoning, chain-of-thought)."""

    type: Literal["thought"] = "thought"
    text: str
    signature: Optional[str] = None  # Base64 encoded signature (GenAI)


class MediaContent(BaseModel):
    """
    Медиа-контент (изображения, аудио, видео, документы).
    Тип конкретного файла определяется внутри Asset.type.
    """

    type: Literal["media"] = "media"
    assets: list[Asset]


class ToolCallContent(BaseModel):
    """Вызов инструмента."""

    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ToolResultContent(BaseModel):
    """Результат выполнения инструмента."""

    type: Literal["tool_result"] = "tool_result"
    tool_result: ToolResult
    assets: Optional[list[Asset]] = None  # Если инструмент вернул файлы


# Объединение всех типов контента (Discriminated Union по полю "type")
ContentItem = (
    TextContent | ThoughtContent | MediaContent | ToolCallContent | ToolResultContent
)


# ══════════════════════════════════════════════════════════════════════════════
# METADATA (метаданные генерации)
# ══════════════════════════════════════════════════════════════════════════════


class UsageStats(BaseModel):
    """Статистика использования токенов."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class MessageMetadata(BaseModel):
    """
    Метаданные генерации (только для role: "assistant").
    """

    model: str  # gpt-4o, gemini-2.0-flash и т.д.
    model_class: Literal["openai", "genai"]  # Тип провайдера
    usage: Optional[UsageStats] = None
    finish_reason: Optional[
        Literal["stop", "tool_calls", "length", "content_filter"]
    ] = None
    latency_ms: Optional[int] = None  # Время генерации


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE (сообщение)
# ══════════════════════════════════════════════════════════════════════════════


class Message(BaseModel):
    """
    Сообщение в истории чата.

    Одна и та же роль не может идти подряд (требование Gemini).
    Если модель возвращает thought → text → tool_call, они собираются в один блок.
    """

    id: str  # Уникальный ID сообщения (msg_xxx)
    timestamp: datetime  # Время создания (ISO 8601)
    role: Literal["system", "user", "assistant", "tool"]
    name: Optional[str] = None  # Имя участника (для multi-user)
    content: list[ContentItem] = Field(default_factory=list)
    metadata: Optional[MessageMetadata] = None  # Только для assistant


# ══════════════════════════════════════════════════════════════════════════════
# CHAT METADATA (настройки чата)
# ══════════════════════════════════════════════════════════════════════════════


class GenAICacheRef(BaseModel):
    """Ссылка на кэшированный контент в Google GenAI."""

    name: str  # cachedContents/xxx
    expires_at: datetime


class CacheRef(BaseModel):
    """Ссылки на кэшированный контент."""

    genai: Optional[GenAICacheRef] = None


class ChatConfig(BaseModel):
    """Конфигурация чата."""

    thinking_mode: Literal["interleaved", "preserved"] = "interleaved"
    provider: Literal["openai", "genai"] = "openai"
    # Можно расширять: model, temperature, max_tokens и т.д.


class ChatMetadata(BaseModel):
    """Метаданные и настройки всего чата."""

    version: Literal["1.0"] = "1.0"  # Версия формата УФС
    config: ChatConfig = Field(default_factory=ChatConfig)
    cache_ref: Optional[CacheRef] = None


# ══════════════════════════════════════════════════════════════════════════════
# CHAT DATA (корневой контейнер)
# ══════════════════════════════════════════════════════════════════════════════


class ChatData(BaseModel):
    """
    Корневая структура УФС — полная история чата.

    Пример использования:
    ```python
    chat = ChatData(
        chat_metadata=ChatMetadata(
            config=ChatConfig(thinking_mode="preserved", provider="genai")
        ),
        messages=[
            Message(
                id="msg_001",
                timestamp=datetime.now(),
                role="user",
                content=[TextContent(text="Привет!")]
            )
        ]
    )
    ```
    """

    chat_metadata: ChatMetadata = Field(default_factory=ChatMetadata)
    messages: list[Message] = Field(default_factory=list)
