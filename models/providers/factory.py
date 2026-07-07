"""
Фабрика моделей: главная точка входа прикладного кода.

make_model("gemini-3.1-flash-lite-preview") -> готовый Provider, собранный
из каталожной спеки и нужных компонентов (конвертер/ассеты/раннер/политика).
"""

from ..base import BaseModel
from ..catalog import get_spec
from .genai import GenaiProvider
from .openai import OpenAiProvider


_SUPPORTED_PROVIDERS = {
    "genai": GenaiProvider,
    "openai": OpenAiProvider,
}


def make_model(model_name: str, system_prompt: str = "") -> BaseModel:
    """
    Собрать готовую модель-оркестратор по имени из каталога.

    :param model_name: Имя модели (ключ каталога).
    :param system_prompt: Системный промпт.
    :return: Экземпляр Provider (GenaiProvider/OpenAiProvider), реализующий BaseModel.
    :raises KeyError: Если модель не зарегистрирована в каталоге.
    :raises UnsupportedProviderError: Если spec.provider неизвестен фабрике.
    """
    spec = get_spec(model_name)
    provider_cls = _SUPPORTED_PROVIDERS.get(spec.provider)
    if provider_cls is None:
        supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Провайдер {spec.provider!r} для модели {model_name!r} не поддерживается. "
            f"Поддерживаемые провайдеры: {supported}"
        )
    return provider_cls(spec=spec, system_prompt=system_prompt)
