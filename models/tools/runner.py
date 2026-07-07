"""
ToolRunner — общий, ПРОВАЙДЕРО-НЕЗАВИСИМЫЙ исполнитель инструментов.

Устраняет дублирование: прежде логика исполнения tool_calls + распаковки
returns_media + сохранения файла была почти дословно скопирована между
GenaiBaseModel.generate и OpenAiBaseModel.generate (~85 строк в каждой).

Работает поверх УФС-типов: на вход — сообщение ассистента с ToolCallContent,
на выход — сообщение роли "tool" с ToolResultContent (и ассетами, если
инструмент вернул медиа). От провайдера ничего не зависит, т.к. вызовы
уже нормализованы в t.ToolCall на стадии converter.from_native.
"""

from typing import Callable, Optional

import utils.types as t
from .media_result import ToolMediaResultBuilder


class ToolRunner:
    """Исполняет вызовы инструментов и упаковывает результаты в УФС."""

    def __init__(self, media_builder: ToolMediaResultBuilder | None = None) -> None:
        """
        :param media_builder: Опциональный билдер для подготовки медиа-результатов инструментов.
        """
        ...

    def run(
        self,
        assistant_message: t.Message,
        tools_executable: dict[str, Callable],
    ) -> Optional[t.Message]:
        """
        Исполнить все tool_call из сообщения ассистента.

        :param assistant_message: Сообщение роли "assistant" с ToolCallContent.
        :param tools_executable: Имя инструмента -> вызываемый объект.
        :return: Сообщение роли "tool" с результатами, либо None, если
                 вызовов инструментов не было.
        """
        ...

    def _execute_one(
        self,
        tool_call: t.ToolCall,
        tools_executable: dict[str, Callable],
    ) -> t.ToolResultContent:
        """
        Исполнить один вызов: найти функцию, вызвать с args, поймать ошибки,
        обработать returns_media (сохранить файл, собрать Asset).

        :param tool_call: Нормализованный вызов инструмента из УФС.
        :param tools_executable: Реестр исполняемых инструментов.
        :return: Результат в виде ToolResultContent (is_error=True при исключении).
        """
        ...

    def _build_media_asset(self, tool: Callable, tool_result) -> t.Asset:
        """
        Распаковать результат media-инструмента (bytes / (text, bytes[, mime]))
        и сохранить файл, вернув заполненный Asset.

        :param tool: Вызванная функция (несёт атрибуты returns_media/mime_type).
        :param tool_result: Сырое возвращённое значение инструмента.
        :return: Asset с local_path, mime_type, size_bytes, data_base64.
        """
        ...
