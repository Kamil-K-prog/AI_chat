"""
Вспомогательные утилиты конвертации для медиа-блоков.

Специфичные для провайдеров конвертеры должны делегировать повторяющееся сопоставление
ассетов с нативными блоками сюда, вместо дублирования логики ветвления inline/cloud-ref в каждой роли.
"""

from typing import Any

import utils.types as t
from ..catalog import ModelSpec


class MediaPartConverter:
    """Формирует фрагменты медиа-данных в нативном формате провайдера из уже подготовленных ассетов УФС."""

    def __init__(self, spec: ModelSpec) -> None:
        """
        :param spec: Выбранная спецификация модели; содержит информацию о возможностях провайдера и медиа.
        """
        ...

    def to_openai_content_part(self, asset: t.Asset) -> dict[str, Any] | None:
        """
        Преобразовать один подготовленный ассет в OpenAI-совместимую часть контента.

        :param asset: Ассет, уже подготовленный соответствующим AssetManager.
        :return: Нативная часть контента или None, если этот ассет следует пропустить.
        """
        ...

    def to_genai_part(self, asset: t.Asset) -> Any:
        """
        Преобразовать один подготовленный ассет в google.genai ``types.Part``.

        :param asset: Ассет, уже подготовленный GenaiAssetManager.
        :return: Часть медиа-контента в нативном формате провайдера.
        """
        ...

    def to_genai_function_response_part(self, asset: t.Asset) -> Any:
        """
        Преобразовать ассет результата работы инструмента в часть GenAI function_response.

        :param asset: Ассет, прикреплённый к ToolResultContent.
        :return: Нативная часть медиа-данных для ответа функции провайдера.
        """
        ...
