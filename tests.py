from inspect import signature, Parameter

from utils.tools import bar_func
from utils.tools_parser import ToolsParser

from pprint import pprint

sig = signature(bar_func)
print(sig.return_annotation)
for number, (name, param) in enumerate(sig.parameters.items()):
    print(
        f"Number - {number}; name - {name}; param - {param}; param.kind - {param.kind}; param.annotation - {param.annotation}; param.default - {param.default}"
    )


pprint(ToolsParser.get_json_schema_openai())

# ========== Тестовый вызов GenAI ==========
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Загружаем переменные окружения
load_dotenv()

# Создаём клиент GenAI
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Получаем схему инструментов для GenAI
tools_schema = ToolsParser.get_types_schema_genai()

# Оборачиваем в types.Tool для передачи в API
tools = [types.Tool(function_declarations=tools_schema)]

print("\n" + "=" * 50)
print("GenAI Tools Schema:")
print("=" * 50)
pprint(tools_schema)

# Тестовый запрос к модели
print("\n" + "=" * 50)
print("Отправляем тестовый запрос к gemini-flash-lite...")
print("=" * 50)

response = client.models.generate_content(
    model="gemini-flash-lite-latest",
    contents="Сложи числа 5.5 и 10 с помощью функции bar_func",
    config=types.GenerateContentConfig(
        tools=tools,
        # Автоматический вызов инструментов отключен для наглядности
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
    ),
)

print("\nОтвет модели:")
print("-" * 50)

# Проверяем, есть ли вызов функции в ответе
for candidate in response.candidates:
    for part in candidate.content.parts:
        if part.function_call:
            print(f"Модель вызвала функцию: {part.function_call.name}")
            print(f"Аргументы: {part.function_call.args}")
        elif part.text:
            print(f"Текст: {part.text}")
