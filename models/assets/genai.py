"""
AssetManager для GenAI (Google Files API).

Переносит сюда логику прежнего GenaiBaseModel._process_asset:
крупные файлы (>20 МБ) заливаются через client.files.upload, для видео/аудио
ожидается состояние ACTIVE, а filename/uri/expires_at пишутся в
asset.cloud_refs.genai для переиспользования.
"""

import utils.types as t
from .base import AssetManager


class GenaiAssetManager(AssetManager):
    """Готовит ассеты к отправке в GenAI: грузит крупные файлы в Files API."""

    def __init__(self, client) -> None:
        """
        :param client: google.genai.Client (для files.upload / files.get).
        """
        ...

    def prepare(self, history: t.ChatData) -> t.ChatData:
        """Пройти по истории и подготовить все ассеты (см. AssetManager.prepare)."""
        ...

    def prepare_asset(self, asset: t.Asset) -> t.Asset:
        """
        Загрузить ассет в Google Files API (или переиспользовать уже
        загруженный по cloud_refs.genai.filename) и записать ссылки обратно.

        Для крупных видео/аудио дожидается выхода файла из состояния PROCESSING.

        :param asset: Ассет УФС.
        :return: Тот же ассет с заполненным cloud_refs.genai.
        """
        ...
