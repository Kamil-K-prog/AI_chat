"""
Каталог моделей провайдера GenAI (Google Gemini).

Каждая модель — это ЗАПИСЬ-ДАННЫЕ (экземпляр GenaiModelSpec), а не класс.
Добавить новую модель Gemini = добавить сюда ещё один экземпляр в GENAI_MODELS.
"""

from ..base import ModelSpec
from ..capabilities import MediaCapabilities, ModelCapabilities, ReasoningCapabilities


class GenaiModelSpec(ModelSpec):
    """
    Спека модели GenAI. Фиксирует provider="genai" и добавляет
    специфичные для GenAI поля при необходимости.
    """

    provider: str = "genai"


# ── Реестр известных моделей GenAI ────────────────────────────────────────────
# Ключ — имя модели, по которому её достаёт фабрика каталога.

GENAI_MODELS: dict[str, GenaiModelSpec] = {
    "gemini-3.1-flash-lite-preview": GenaiModelSpec(
        name="gemini-3.1-flash-lite-preview",
        display_name="Gemini 3.1 Flash Lite",
        capabilities=ModelCapabilities(
            reasoning=ReasoningCapabilities(supported=True, include_thoughts=True, effort="medium"),
            media=MediaCapabilities(vision=True, audio=True, video=True, documents=True),
        ),
    ),
}
