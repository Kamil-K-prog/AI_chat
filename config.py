# --- AI generated file ---

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Класс настроек приложения.
    Pydantic автоматически подтянет значения из переменных окружения
    или из .env файла, если они там определены.
    """

    # Ключи API для различных сервисов
    GEMINI_API_KEY: str
    OPENAI_API_KEY: str
    DEEPSEEK_API_KEY: str
    GLM_API_KEY: str

    # Настройки OpenRouter
    OPENROUTER_API_KEY: str
    OPENROUTER_API_SECRET: str
    OPENROUTER_MODEL: str = "xiaomi/mimo-v2-flash:free"

    # Системный промпт
    SYSTEM_PROMPT: str

    # Конфигурация Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",  # Путь к файлу с переменными окружения
        env_file_encoding="utf-8",
        extra="ignore",  # Игнорировать лишние переменные в .env
    )


# Создаем глобальный экземпляр настроек
settings = Settings()
