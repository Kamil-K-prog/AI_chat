# DeepSeek Model
# TODO: Реализовать класс для моделей DeepSeek

from models import OpenAiBaseModel
from config import settings


class DeepseekReasoner(OpenAiBaseModel):
    def __init__(self,
                 model_name="deepseek-reasoner",
                 system_prompt="Ты полезный ИИ ассистент",
                 base_url="https://api.deepseek.com",
                 api_key=settings.DEEPSEEK_API_KEY):
        super().__init__(model_name, system_prompt, base_url, api_key, True)


class DeepseekChat(OpenAiBaseModel):
    def __init__(self,
                 model_name="deepseek-chat",
                 system_prompt="Ты полезный ИИ ассистент",
                 base_url="https://api.deepseek.com",
                 api_key=settings.DEEPSEEK_API_KEY):
        super().__init__(model_name, system_prompt, base_url, api_key, False)

