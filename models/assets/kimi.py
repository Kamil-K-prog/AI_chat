"""
AssetManager для Kimi (Moonshot).

Особенности Kimi, которые здесь инкапсулированы:
  - Изображения/видео <20 МБ идут инлайном (data:base64) — это формирует
    конвертер, здесь подготовка не нужна.
  - Крупные файлы заливаются через client.files.create -> ms://<id>,
    и id пишется обратно в asset.cloud_refs (ИСПРАВЛЕНИЕ прежнего бага:
    раньше результат загрузки никуда не сохранялся и грузился заново).
  - Документы передаются отдельным system-сообщением: текст извлекается
    здесь (files.create purpose="file-extract" -> files.content) и пишется
    в asset.ocr_text. Само превращение ocr_text в system-сообщение —
    задача конвертера, не загрузчика.
"""

import utils.types as t
from .base import AssetManager


class KimiAssetManager(AssetManager):
    """Готовит ассеты к отправке в Kimi: заливка крупных файлов и OCR документов."""

    def __init__(self, client) -> None:
        """
        :param client: OpenAI-совместимый клиент Moonshot (для files.create / files.content).
        """
        ...

    def prepare(self, history: t.ChatData) -> t.ChatData:
        """Пройти по истории и подготовить все ассеты (см. AssetManager.prepare)."""
        ...

    def prepare_asset(self, asset: t.Asset) -> t.Asset:
        """
        Подготовить ассет под Kimi:
          - крупное изображение/видео -> files.create, записать ms://id в cloud_refs;
          - документ -> извлечь текст в asset.ocr_text (если ещё не извлечён);
          - мелкое медиа -> ничего (пойдёт инлайном через конвертер).

        :param asset: Ассет УФС.
        :return: Тот же ассет с дозаполненными полями.
        """
        ...
