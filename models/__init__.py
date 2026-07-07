"""Публичный API нового слоя моделей."""

from .base import BaseModel
from .providers import GenaiProvider, OpenAiProvider, make_model
from .catalog import ModelSpec, get_spec, list_specs
from .history import HistoryAppender, HistoryValidator
from .request import GenerationRequestOptions

__all__ = [
    "BaseModel",
    "GenaiProvider",
    "OpenAiProvider",
    "make_model",
    "ModelSpec",
    "get_spec",
    "list_specs",
    "HistoryAppender",
    "HistoryValidator",
    "GenerationRequestOptions",
]