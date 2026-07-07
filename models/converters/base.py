"""
HistoryConverter — ЧИСТАЯ конвертация между УФС и нативным форматом провайдера.

Ключевой принцип: здесь НЕТ сетевых вызовов и НЕТ загрузки файлов.
Конвертер читает уже подготовленные данные (asset.cloud_refs / asset.ocr_text,
проставленные AssetManager на предыдущей стадии конвейера) и только формирует
структуру под конкретный API. Благодаря этому конвертер детерминирован и
тестируется без обращения к сети.

Ответственности:
  - to_native:   УФС-история  -> нативный формат запроса провайдера
  - from_native: сырой ответ провайдера -> список Message в УФС (delta ассистента)
"""

from abc import ABC, abstractmethod
from typing import Any
import utils.types as t
from ..catalog import ModelSpec


class HistoryConverter(ABC):
    """
    Базовый интерфейс конвертера истории.

    spec прокидывается, чтобы конвертер знал возможности модели
    (например, включать ли мысли в запрос при reasoning-моделях).
    """

    def __init__(self, spec: ModelSpec) -> None:
        ...

    @abstractmethod
    def to_native(self, history: t.ChatData) -> Any:
        """
        Конвертировать УФС-историю в нативный формат запроса провайдера.

        Предполагается, что медиа уже подготовлено AssetManager: для крупных
        файлов в asset.cloud_refs лежат ссылки, для документов — asset.ocr_text.
        Сеть здесь не используется.

        :param history: История чата в УФС.
        :return: Нативное представление (тип зависит от провайдера).
        """
        ...

    @abstractmethod
    def from_native(self, response: Any) -> list[t.Message]:
        """
        Разобрать сырой ответ провайдера в дельту сообщений УФС.

        Возвращает только сообщение(я) ассистента (text/thought/tool_call/media).
        Исполнение инструментов сюда НЕ входит — этим занимается ToolRunner.

        :param response: Сырой объект ответа от SDK провайдера.
        :return: Список Message (обычно одно сообщение роли "assistant").
        """
        ...
