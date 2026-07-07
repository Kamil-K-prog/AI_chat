"""
Каталог моделей OpenAI-совместимых провайдеров (DeepSeek, Kimi, OpenRouter и т.д.).

Все они говорят по одному протоколу (chat.completions), поэтому отличаются
только ДАННЫМИ: base_url, имя ключа в настройках, имя модели и возможности.
Поэтому здесь — несколько спек, сгруппированных по вендору, но один класс спеки.

Добавить новую модель = добавить экземпляр в соответствующий dict.
Новый OpenAI-совместимый вендор = ещё один dict + запись в OPENAI_MODELS.
"""

from typing import Optional

from ..base import ModelSpec
from ..capabilities import MediaCapabilities, ModelCapabilities, ReasoningCapabilities


class OpenAiModelSpec(ModelSpec):
    """
    Спека OpenAI-совместимой модели.

    Добавляет транспортные данные, общие для всех OpenAI-совместимых вендоров:
    адрес API и имя ключа в settings (не сам ключ — секреты не храним в каталоге).
    """

    provider: str = "openai"
    base_url: Optional[str] = None  # Адрес API вендора
    api_key_setting: str = "OPENAI_API_KEY"  # Имя поля в config.settings, откуда брать ключ


# ── DeepSeek ──────────────────────────────────────────────────────────────────

DEEPSEEK_MODELS: dict[str, OpenAiModelSpec] = {
    "deepseek-v4-flash": OpenAiModelSpec(
        name="deepseek-v4-flash",
        display_name="DeepSeek Chat",
        base_url="https://api.deepseek.com",
        api_key_setting="DEEPSEEK_API_KEY",
    ),
    "deepseek-v4-pro": OpenAiModelSpec(
        name="deepseek-v4-pro",
        display_name="DeepSeek Reasoner",
        base_url="https://api.deepseek.com",
        api_key_setting="DEEPSEEK_API_KEY",
        capabilities=ModelCapabilities(
            reasoning=ReasoningCapabilities(supported=True, include_thoughts=True, effort="medium"),
        ),
    ),
}


# ── Kimi (Moonshot) ───────────────────────────────────────────────────────────

KIMI_MODELS: dict[str, OpenAiModelSpec] = {
    "kimi-k2.6": OpenAiModelSpec(
        name="kimi-k2.6",
        display_name="Kimi K2.6",
        base_url="https://api.moonshot.ai/v1",
        api_key_setting="KIMI_API_KEY",
        capabilities=ModelCapabilities(
            reasoning=ReasoningCapabilities(supported=True),
            media=MediaCapabilities(vision=True, video=True, documents=True),
        ),
    ),
}


# Сводный реестр всех OpenAI-совместимых моделей.
OPENAI_MODELS: dict[str, OpenAiModelSpec] = {
    **DEEPSEEK_MODELS,
    **KIMI_MODELS,
}
