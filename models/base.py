"""
Базовые интерфейсы слоя моделей.

Provider — оркестратор: он НЕ конвертирует, НЕ грузит файлы и НЕ исполняет
инструменты сам. Он связывает (композиция/DI) специализированные компоненты:

    history = asset_manager.prepare(history)      # I/O: загрузка, OCR, cloud_refs
    history = thinking_policy.apply(history)      # чистая трансформация УФС
    native  = converter.to_native(history)        # чистая конвертация (без сети)
    response = self._do_request(native, tools)    # сам вызов API (на провайдере)
    delta    = converter.from_native(response)    # разбор ответа в УФС
    tool_msg = tool_runner.run(delta[-1], execs)  # исполнение инструментов + медиа

Так каждая ответственность живёт в своём классе (SRP), а конкретные модели
выражаются ДАННЫМИ (ModelSpec из каталога), а не подклассами.
"""

from abc import ABC, abstractmethod
from typing import Callable

import utils.types as t
from .request import GenerationRequestOptions


class BaseModel(ABC):
    """
    Сохраняем прежний публичный контракт generate(), чтобы test.py и
    вызывающий код продолжали работать без изменений.
    """

    @abstractmethod
    def generate(
        self,
        history: t.ChatData,
        tools_definition,
        tools_executable: dict[str, Callable],
        options: GenerationRequestOptions | None = None,
        react_loop: bool = False,
    ) -> tuple[t.ChatData, list[t.Message]]:
        """
        Сделать шаг генерации: подготовить историю, вызвать API, разобрать ответ.
        В зависимости от флага react_loop, либо выполнить ровно один шаг с инструментами
        (пошаговый режим), либо крутить цикл вызова модели и инструментов до получения
        финального текстового ответа (автоматический режим).

        :param history: История чата в УФС.
        :param tools_definition: Схема инструментов в нативном формате провайдера.
        :param tools_executable: Имя инструмента -> вызываемый объект.
        :param options: Нормализованные параметры генерации (температура, токены и т.д.).
        :param react_loop: True, чтобы автоматически выполнять циклы вызова тулов до получения текста.
        :return: Кортеж (обновлённая история, список новых сообщений-дельта).
        """
        
