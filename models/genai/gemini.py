# Gemini Model
# TODO: Реализовать класс для моделей Google Gemini

from models import GenaiBaseModel

class Gemini3_1FlashLite(GenaiBaseModel):
    def __init__(self, is_reasoning=True, include_thoughts=True, reasoning_effort="medium", system_prompt=None):
        super().__init__("gemini-3.1-flash-lite-preview", is_reasoning, include_thoughts, reasoning_effort, system_prompt)
        