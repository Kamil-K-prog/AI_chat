"""
NullAssetManager — Null Object для немультимодальных моделей (напр. DeepSeek Chat).

Ничего не загружает и не меняет: просто возвращает историю как есть.
Позволяет оркестратору всегда вызывать asset_manager.prepare(...) без
проверок "а поддерживает ли эта модель медиа" (убирает if-ы по возможностям).
"""

import utils.types as t
from .base import AssetManager


class NullAssetManager(AssetManager):
    """Заглушка: ассеты не требуют подготовки."""

    def prepare(self, history: t.ChatData) -> t.ChatData:
        """Вернуть историю без изменений."""
        return history

    def prepare_asset(self, asset: t.Asset) -> t.Asset:
        """Вернуть ассет без изменений."""
        return asset
