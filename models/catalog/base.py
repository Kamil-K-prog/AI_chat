"""
Базовые data-объекты каталога моделей.

ModelSpec — это ЧИСТЫЕ ДАННЫЕ (Pydantic), описывающие одну конкретную LLM:
её кодовое имя у провайдера и набор возможностей (reasoning, vision и т.д.).

Смысл каталога: добавление новой модели = добавление записи-данных,
а НЕ написание нового класса. Это Open/Closed: поведение (Provider/Converter)
закрыто для изменения, расширяемся данными (ModelSpec).

При смене модели "на лету" подменяется ВЕСЬ ModelSpec целиком, поэтому
возможности (reasoning/vision) всегда едут вместе с именем модели и не
рассинхронизируются.
"""

from typing import Optional

from pydantic import BaseModel

from .capabilities import ModelCapabilities


class ModelSpec(BaseModel):
    """
    Провайдеро-независимое описание одной LLM.

    Провайдеро-специфичные поля (base_url, ключи, особые флаги) добавляются
    в подклассах внутри catalog/<provider>.py.
    """

    name: str  # Кодовое имя модели у провайдера (напр. "gemini-3.1-flash-lite-preview")
    # К какому протоколу/оркестратору относится модель. Допустимые значения: "genai", "openai".
    # Тип намеренно str (а не Literal), чтобы подклассы-каталоги могли зафиксировать
    # конкретное значение без конфликта инвариантности mutable-поля Pydantic.
    provider: str

    display_name: Optional[str] = None  # Человекочитаемое имя для UI
    capabilities: ModelCapabilities = ModelCapabilities()
