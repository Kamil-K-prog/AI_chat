from google.genai import types
from google import genai
from google.genai.types import GenerateContentResponse
from typing import List, Dict, Callable

from .base_model import BaseModel

from config import settings


class GenaiBaseModel(BaseModel):
    """Базовая модель, наследники будут только изменять что-то под конкретные типы genai моделей - уровни ризонинга, кодовое имя модели...

    Благодаря унификации все модели принимают историю сообщений в формате-адаптере, затем конвертируют его в нужный для себя формат, делают запрос, конвертируют обратно и возвращают.
    """

    def __init__(self):
        self.model_name = None  # Будет переопределено в конкретном классе модели
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _do_request(self, messages: List[types.Content]) -> GenerateContentResponse:
        """Совершает единичный запрос к API модели"""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=messages,
            config=types.GenerateContentConfig(
                tools=self.tools_definition,
            ),
        )

    def generate(self, messages, tools_definition, tools_executable: Dict[str, Callable]) -> List[List, List[types.Content]]:
        """Генерирует ответ на сообщение. Возвращает:
        1) Дельту (сообщения, которые были сгенерированы в этом раунде
        2) Новую, дополненную историю сообщений
        """
        pass
