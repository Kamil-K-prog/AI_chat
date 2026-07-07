"""
Объекты данных о возможностях (capabilities) для записей каталога моделей.

Возможности сгруппированы отдельно от ModelSpec, чтобы по мере развития протоколов
каталог не превращался в длинный список слабо связанных логических (boolean) полей.
"""

from typing import Literal

from pydantic import BaseModel


class ReasoningCapabilities(BaseModel):
    """Описывает, поддерживает ли модель вывод мыслей (reasoning/thinking) и в каком формате."""

    supported: bool = False
    include_thoughts: bool = True
    effort: Literal["low", "medium", "high"] = "medium"


class MediaCapabilities(BaseModel):
    """Описывает, какие категории входящих медиафайлов может принимать модель."""

    vision: bool = False
    audio: bool = False
    video: bool = False
    documents: bool = False


class ToolCapabilities(BaseModel):
    """Описывает возможности вызова инструментов/функций (tool/function-calling), предоставляемые эндпоинтом модели."""

    tool_calling: bool = True
    parallel_tool_calls: bool = False
    strict_schema: bool = False


class ResponseCapabilities(BaseModel):
    """Описывает возможности структурированного вывода и стриминга эндпоинта модели."""

    streaming: bool = False
    json_mode: bool = False
    structured_output: bool = False


class ModelCapabilities(BaseModel):
    """Объединяет все группы возможностей для одной записи модели в каталоге."""

    reasoning: ReasoningCapabilities = ReasoningCapabilities()
    media: MediaCapabilities = MediaCapabilities()
    tools: ToolCapabilities = ToolCapabilities()
    response: ResponseCapabilities = ResponseCapabilities()
