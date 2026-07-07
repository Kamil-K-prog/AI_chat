"""
Выбор компонентов для сборки провайдера.

Этот модуль избавляет файлы провайдеров и фабрик от накопления логики if/else о том,
какой конвертер, менеджер ассетов или реализация политики соответствуют конкретному ModelSpec.
"""

from openai import OpenAI
from google import genai

from ..assets import AssetManager
from ..catalog import ModelSpec
from ..converters import HistoryConverter
from ..thinking import ThinkingPolicy
from ..tools import ToolRunner


class ProviderComponentSelector:
    """Выбирает конкретных участников (collaborators) для провайдера на основе спецификации модели."""

    def build_openai_converter(self, spec: ModelSpec) -> HistoryConverter:
        """
        Выбрать OpenAI-совместимый конвертер истории для ``spec``.

        :param spec: Запись в каталоге для выбранной OpenAI-совместимой модели.
        :return: Конвертер, сопоставляющий структуры УФС со структурами chat.completions.
        """
        ...

    def build_genai_converter(self, spec: ModelSpec) -> HistoryConverter:
        """
        Выбрать конвертер истории Google GenAI для ``spec``.

        :param spec: Запись в каталоге для выбранной модели GenAI.
        :return: Конвертер, сопоставляющий структуры УФС со структурами google.genai.
        """
        ...

    def build_openai_asset_manager(self, spec: ModelSpec, client: OpenAI) -> AssetManager:
        """
        Выбрать стратегию подготовки ассетов для OpenAI-совместимой модели.

        :param spec: Запись каталога, содержащая возможности и настройки вендора.
        :param client: OpenAI-совместимый клиент, используемый менеджерами для обращения к API файлов.
        :return: Менеджер ассетов (AssetManager), подходящий для данной модели/вендора.
        """
        ...

    def build_genai_asset_manager(self, spec: ModelSpec, client: genai.Client) -> AssetManager:
        """
        Выбрать стратегию подготовки ассетов для модели Google GenAI.

        :param spec: Запись каталога, содержащая возможности работы с медиа.
        :param client: Клиент Google GenAI, используемый для работы с Files API.
        :return: Менеджер ассетов (AssetManager), подходящий для GenAI.
        """
        ...

    def build_tool_runner(self, spec: ModelSpec) -> ToolRunner:
        """
        Выбрать провайдеро-независимый исполнитель инструментов (ToolRunner).

        :param spec: Запись в каталоге; доступна для будущих вариантов исполнения инструментов.
        :return: Экземпляр ToolRunner.
        """
        ...

    def build_thinking_policy(self, spec: ModelSpec) -> ThinkingPolicy:
        """
        Выбрать политику, преобразующую контент рассуждений (reasoning) перед конвертацией.

        :param spec: Запись в каталоге; доступна для будущих специфичных для провайдера или модели политик.
        :return: Экземпляр ThinkingPolicy.
        """
        ...
