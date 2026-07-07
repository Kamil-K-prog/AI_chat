"""
GenaiProvider — оркестратор для моделей Google GenAI.

Связывает GenaiHistoryConverter, GenaiAssetManager, ToolRunner, ThinkingPolicy
и сам делает запрос через google.genai. Конкретная модель задаётся ModelSpec.
"""

from typing import Callable

from google.genai import types
import utils.types as t
from ..base import BaseModel
from ..catalog import ModelSpec
from ..converters import GenaiHistoryConverter
from ..assets import GenaiAssetManager
from ..tools import ToolRunner
from ..thinking import ThinkingPolicy
from ..history import HistoryAppender
from ..request import GenerationRequestOptions


class GenaiProvider(BaseModel):
    """Оркестратор запросов к GenAI для произвольной модели из каталога."""

    def __init__(
        self,
        spec: ModelSpec,
        system_prompt: str = "",
        converter: GenaiHistoryConverter | None = None,
        asset_manager: GenaiAssetManager | None = None,
        tool_runner: ToolRunner | None = None,
        thinking_policy: ThinkingPolicy | None = None,
        history_appender: HistoryAppender | None = None,
    ) -> None:
        """
        :param spec: Спека модели из каталога (имя, reasoning, vision, ...).
        :param system_prompt: Системный промпт.
        :param converter: Конвертер истории (по умолчанию GenaiHistoryConverter(spec)).
        :param asset_manager: Менеджер ассетов (по умолчанию GenaiAssetManager).
        :param tool_runner: Исполнитель инструментов (по умолчанию общий ToolRunner).
        :param thinking_policy: Политика мыслей (по умолчанию общая ThinkingPolicy).
        :param history_appender: Аппендер для записи дельта в историю.
        """
        ...

    def _build_thinking_config(self) -> types.ThinkingConfig | None:
        """
        Собрать ThinkingConfig из spec.capabilities.reasoning
        либо None для не-reasoning моделей.

        :return: types.ThinkingConfig или None.
        """
        ...

    def _do_request(
        self,
        native_history,
        tools_definition,
        options: GenerationRequestOptions | None = None,
    ):
        """
        Выполнить запрос client.models.generate_content.

        :param native_history: Список types.Content.
        :param tools_definition: Инструменты в схеме GenAI.
        :param options: Нормализованные параметры генерации.
        :return: Сырой GenerateContentResponse.
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
