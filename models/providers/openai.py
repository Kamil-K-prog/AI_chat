"""
OpenAiProvider — оркестратор для всех OpenAI-совместimых вендоров
(DeepSeek, Kimi, OpenRouter, ...).

Один класс на весь протокол: вендоры отличаются данными (base_url, ключ,
имя модели, возможности) из ModelSpec, а не кодом. Мультимодальная специфика
(напр. загрузка у Kimi) инкапсулирована в подменяемом AssetManager.
"""

from typing import Callable

import utils.types as t
from ..base import BaseModel
from ..catalog import ModelSpec
from ..converters import OpenAiHistoryConverter
from ..assets import AssetManager
from ..tools import ToolRunner
from ..thinking import ThinkingPolicy
from ..history import HistoryAppender
from ..request import GenerationRequestOptions


class OpenAiProvider(BaseModel):
    """Оркестратор запросов к OpenAI-совместимым API для модели из каталога."""

    def __init__(
        self,
        spec: ModelSpec,
        system_prompt: str = "",
        converter: OpenAiHistoryConverter | None = None,
        asset_manager: AssetManager | None = None,
        tool_runner: ToolRunner | None = None,
        thinking_policy: ThinkingPolicy | None = None,
        history_appender: HistoryAppender | None = None,
    ) -> None:
        """
        :param spec: Спека модели (несёт base_url, api_key_setting, возможности).
        :param system_prompt: Системный промпт.
        :param converter: Конвертер истории (по умолчанию OpenAiHistoryConverter(spec)).
        :param asset_manager: Менеджер ассетов; для немультимодальных — NullAssetManager,
                              для Kimi — KimiAssetManager. По умолчанию подбирается по spec.
        :param tool_runner: Исполнитель инструментов (по умолчанию общий ToolRunner).
        :param thinking_policy: Политика мыслей (по умолчанию общая ThinkingPolicy).
        :param history_appender: Аппендер для записи дельта в историю.
        """
        ...

    def _create_client(self):
        """
        Создать OpenAI-клиент из spec.base_url и ключа (по spec.api_key_setting
        из config.settings).

        :return: Экземпляр openai.OpenAI.
        """
        ...

    def _do_request(
        self,
        native_history,
        tools_definition,
        options: GenerationRequestOptions | None = None,
        extra_body=None,
    ):
        """
        Выполнить запрос client.chat.completions.create.

        :param native_history: Список message-словарей.
        :param tools_definition: Инструменты в JSON-схеме OpenAI.
        :param options: Нормализованные параметры генерации.
        :param extra_body: Доп. параметры запроса.
        :return: Сырой ChatCompletion.
        """
        ...

    def generate(
        self,
        history: t.ChatData,
        tools_definition,
        tools_executable: dict[str, Callable],
        options: GenerationRequestOptions | None = None,
        react_loop: bool = False,
    ) -> tuple[t.ChatData, list[t.Message]]:
        """Шаг генерации по общему конвейеру (см. BaseModel.generate)."""
        ...
