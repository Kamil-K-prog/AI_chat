"""
Конвертер истории для OpenAI-совместимых провайдеров (DeepSeek, Kimi, ...).

Переносит сюда ЧИСТУЮ часть прежнего OpenAiBaseModel._convert_history_from_umf
и разбора response.choices[0].message — без сетевой загрузки ассетов.

Медиа-ассеты превращаются в нативные блоки на основе УФС-данных
(data_base64 / cloud_refs), которые проставил AssetManager. Документы,
которые у некоторых вендоров (Kimi) подаются отдельным system-сообщением,
формируются здесь из готового asset.ocr_text.
"""

from typing import Any
import utils.types as t
from .base import HistoryConverter


class OpenAiHistoryConverter(HistoryConverter):
    """Конвертер УФС <-> список message-словарей chat.completions."""

    def to_native(self, history: t.ChatData) -> list[dict]:
        """
        Конвертировать УФС в список message-словарей для chat.completions.

        :param history: История чата в УФС.
        :return: Список словарей {"role": ..., "content": ..., ...}.
        """
        ...

    def from_native(self, response: Any) -> list[t.Message]:
        """
        Разобрать ChatCompletion в дельту ассистента УФС.

        Разбирает message.reasoning_content (если reasoning), message.content,
        message.tool_calls.

        :param response: Ответ client.chat.completions.create.
        :return: Список Message с одним сообщением роли "assistant".
        """
        ...
