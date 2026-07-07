"""Объекты параметров запроса, используемые оркестраторами провайдеров."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationRequestOptions:
    """Нормализованные параметры для одного шага генерации перед провайдеро-специфичным маппингом."""

    extra_body: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class RequestOptionsMapper:
    """Сопоставляет нормализованные параметры генерации с нативными kwargs запроса провайдера."""

    def to_openai_kwargs(self, options: GenerationRequestOptions) -> dict[str, Any]:
        """
        Преобразовать нормализованные параметры в именованные аргументы (kwargs) для ``chat.completions.create``.

        :param options: Провайдеро-независимые параметры генерации.
        :return: Именованные аргументы для OpenAI-совместимого запроса.
        """
        ...

    def to_genai_config_kwargs(self, options: GenerationRequestOptions) -> dict[str, Any]:
        """
        Преобразовать нормализованные параметры в kwargs для GenAI ``GenerateContentConfig``.

        :param options: Провайдеро-независимые параметры генерации.
        :return: Именованные аргументы для создания конфигурации запроса GenAI.
        """
        ...
