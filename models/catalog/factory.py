"""
Фабрика/реестр каталога моделей.

Единая точка входа: по имени модели вернуть её ModelSpec.
Здесь сводятся реестры всех провайдеров. Добавление нового провайдера =
импортировать его реестр и подмешать в _ALL_MODELS.
"""

from .base import ModelSpec
from .providers.genai import GENAI_MODELS
from .providers.openai import OPENAI_MODELS


# Сводный реестр всех известных моделей: имя -> ModelSpec.
_ALL_MODELS: dict[str, ModelSpec] = {
    **GENAI_MODELS,
    **OPENAI_MODELS,
}


def get_spec(model_name: str) -> ModelSpec:
    """
    Вернуть ModelSpec по имени модели.

    :param model_name: Кодовое имя модели (ключ в каталоге).
    :return: Соответствующая спека.
    :raises KeyError: Если модель не зарегистрирована в каталоге.
    """
    try:
        return _ALL_MODELS[model_name]
    except KeyError as exc:
        known_models = ", ".join(sorted(_ALL_MODELS))
        raise KeyError(
            f"Модель {model_name!r} не зарегистрирована в каталоге. "
            f"Доступные модели: {known_models}"
        ) from exc


def list_specs() -> list[ModelSpec]:
    """
    Вернуть список всех зарегистрированных спек (для UI/выбора модели).

    :return: Список ModelSpec из всех реестров провайдеров.
    """
    return list(_ALL_MODELS.values())
