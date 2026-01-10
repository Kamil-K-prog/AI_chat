# --- AI generated file ---

from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field

# --- 1. Базовые кирпичики контента ---

class TextData(BaseModel):
    type: Literal["text"] = "text"
    data: str

class ThoughtsData(BaseModel):
    type: Literal["thoughts"] = "thoughts"
    data: str

# Для медиа
class MediaItem(BaseModel):
    type: Literal["image", "audio", "video"]
    source_type: Literal["base64", "url", "path"]
    content: str

class MediaData(BaseModel):
    type: Literal["media"] = "media"
    data: List[MediaItem]

# Для инструментов
class ToolCallItem(BaseModel):
    tool_name: str
    id: str
    args: Dict[str, Any]

class ToolCallsData(BaseModel):
    type: Literal["tool_calls"] = "tool_calls"
    data: List[ToolCallItem]

class ToolResponseItem(BaseModel):
    tool_name: str
    id: str
    error: bool = False
    response: str  # Или Dict, если ответ сложный

class ToolResponseData(BaseModel):
    type: Literal["tool_response"] = "tool_response" # В твоем примере это был tool_response
    data: ToolResponseItem

# --- 2. Объединение (Discriminated Union) ---
# Это магия Pydantic: он сам поймет, какой класс использовать,
# посмотрев на поле "type" во входных данных.
ContentItem = Union[
    TextData, 
    ThoughtsData, 
    MediaData, 
    ToolCallsData, 
    ToolResponseData
]

# --- 3. Метаданные ---

class UsageStats(BaseModel):
    input: int
    output: int

class Metadata(BaseModel):
    usage: Optional[UsageStats] = None
    model_name: Optional[str] = None

# --- 4. Главная структура сообщения ---

class Message(BaseModel):
    role: Literal["system", "user", "model", "tools_responses"]
    content: List[ContentItem] = Field(default_factory=list)
    metadata: Optional[Metadata] = None

# --- 5. Внешний контейнер (история) ---
class ChatHistory(BaseModel):
    messages: List[Message] = Field(default_factory=list)