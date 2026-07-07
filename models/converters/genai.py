"""
Конвертер истории для GenAI (Google Gemini).

Переносит сюда ЧИСТУЮ часть прежнего GenaiBaseModel._convert_history_from_umf
и разбора response.candidates[0].content.parts — без вызовов client.files.*
(загрузка теперь в assets/genai.py).

Особое внимание — thought_signature: при reasoning-моделях Gemini 3 подписи
мыслей в активном цикле ReAct нужно возвращать обратно, иначе API вернёт 400.
"""
from pydoc import text
from typing import Any, List
from google.genai import types
import utils.types as t
from utils.types import Asset
from .base import HistoryConverter
from utils.small_utils import (
    message_helper,
    generate_timestamp,
    string_to_bytes,
    bytes_to_string,
    file_to_bytes
)


class GenaiHistoryConverter(HistoryConverter):
    """Конвертер УФС <-> google.genai types.Content."""

    def to_native(self, history: t.ChatData) -> List[types.Content]:
        """
        Конвертировать УФС в список types.Content.

        Маппинг ролей: assistant -> "model", tool -> "user" (function_response),
        system обрабатывается отдельно (system_instruction в конфиге запроса).

        :param history: История чата в УФС.
        :return: Список types.Content для contents= запроса.
        """
        ...

    def from_native(self, response: Any) -> list[t.Message]:
        """
        Разобрать GenerateContentResponse в дельту ассистента УФС.

        Разбирает part.thought (+ thought_signature -> ThoughtContent.signature),
        part.text, part.function_call, part.inline_data (сгенерированное изображение).

        :param response: Ответ client.models.generate_content.
        :return: Список Message с одним сообщением роли "assistant".
        """
        ...

    def _content_to_parts(self, message: t.Message) -> List[types.Part]:
        ...