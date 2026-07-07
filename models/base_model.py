"""
Совместимый мост (shim) для старых импортов.

Старый код, который делает ``from models.base_model import BaseModel``, всё ещё работает —
он перенаправляет на новый класс ``models.base.BaseModel``.

Новый код должен импортировать напрямую из ``models.base``.
Устаревшие реализации (``genai_base_model``, ``openai_base_model``)
остаются в репозитории исключительно в качестве справочного материала.
"""

from .base import BaseModel

__all__ = ["BaseModel"]
