from models.genai.gemini import Gemini3_1FlashLite
import utils.types as t
from utils.small_utils import generate_timestamp
from utils.tools_parser.tools_parser import ToolsParser
import utils.tools


def create_initial_history() -> t.ChatData:
    return t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(provider="genai")),
        messages=[
            t.Message(
                id="user_1",
                timestamp=generate_timestamp(),
                role="user",
                content=[
                    t.TextContent(
                        text="Сложи два числа 5634 и 77854 с помощью bar_func. Не пиши ответ самостоятельно, ОБЯЗАТЕЛЬНО вызови функцию"
                    )
                ],
            ),
        ],
    )


def auto_test(model: Gemini3_1FlashLite, tools_def, tools_exec):
    print("\n--- Запуск автоматического тестирования ---")
    print("Формирование начальной истории УФС...")
    history = create_initial_history()

    print("\n--- ПЕРВЫЙ ВЫЗОВ (ожидаем tool_call) ---")
    history, delta1 = model.generate(
        history=history, tools_definition=tools_def, tools_executable=tools_exec
    )

    print(f"Сгенерировано сообщений: {len(delta1)}")
    for msg in delta1:
        print(f"Роль: {msg.role}")
        for c in msg.content:
            if c.type == "tool_call":
                print(
                    f"  Вызов инструмента: {c.tool_call.name} с аргументами {c.tool_call.args}"
                )
            elif c.type == "tool_result":
                print(
                    f"  Результат инструмента {c.tool_result.name}: {c.tool_result.content} (error: {c.tool_result.is_error})"
                )
            elif c.type == "text":
                print(f"  Текст: {c.text}")
            elif c.type == "thought":
                print(f"  Мысли: {c.text}")

    print("\n--- ВТОРОЙ ВЫЗОВ (ожидаем финальный ответ) ---")

    # Должен учесть результаты инструментов и дать ответ
    history, delta2 = model.generate(
        history=history, tools_definition=tools_def, tools_executable=tools_exec
    )

    print(f"Сгенерировано сообщений: {len(delta2)}")
    print("Отладка: \n", history.model_dump_json(indent=2))
    for msg in delta2:
        print(f"Роль: {msg.role}")
        for c in msg.content:
            if c.type == "text":
                print(f"  Текст: {c.text}")
            elif c.type == "thought":
                print(f"  Мысли: {c.text}")


def chat_mode(model: Gemini3_1FlashLite, tools_def, tools_exec):
    from utils.small_utils.messages_helper import message_helper

    print("\n--- Режим интерактивного чата (для выхода введите 'exit' или 'выход') ---")

    # Инициализируем пустую историю
    history = t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(provider="genai")), messages=[]
    )

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() in ("exit", "выход", "quit"):
            break

        if not user_input:
            continue

        # Добавляем сообщение пользователя в историю
        history.messages.append(
            t.Message(
                id=message_helper.generate_id(9),
                timestamp=generate_timestamp(),
                role="user",
                content=[t.TextContent(text=user_input)],
            )
        )

        while True:
            # Вызываем генерацию
            history, delta = model.generate(
                history=history, tools_definition=tools_def, tools_executable=tools_exec
            )

            has_tool_calls_or_results = False

            # Выводим последнюю дельту
            for msg in delta:
                for c in msg.content:
                    if c.type == "thought":
                        print(f"[Мысли]: {c.text}")
                    elif c.type == "text":
                        print(f"[Ассистент]: {c.text}")
                    elif c.type == "tool_call":
                        has_tool_calls_or_results = True
                        print(
                            f"[Вызов инструмента]: {c.tool_call.name}({c.tool_call.args})"
                        )
                    elif c.type == "tool_result":
                        has_tool_calls_or_results = True
                        print(f"[Результат инструмента]: {c.tool_result.content}")

            # Если были вызовы инструментов (последним добавилось сообщение tool),
            # нужно снова вызвать модель, чтобы она ответила на основе результата
            if has_tool_calls_or_results and delta[-1].role == "tool":
                continue
            else:
                break


def main():
    print("Инициализация модели...")
    model = Gemini3_1FlashLite(
        system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко."
    )

    tools_def = ToolsParser.get_types_schema_genai()
    tools_exec = ToolsParser.get_tools_callables()

    print(f"Доступные инструменты: {list(tools_exec.keys())}")

    while True:
        print("\nВыберите режим:")
        print("1 - Автоматическая проверка (старый тест)")
        print("2 - Интерактивный чат")
        print("0 - Выход")
        choice = input("> ").strip()

        if choice == "1":
            auto_test(model, tools_def, tools_exec)
        elif choice == "2":
            chat_mode(model, tools_def, tools_exec)
        elif choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте еще раз.")


if __name__ == "__main__":
    from config import settings

    # Убедимся, что ключ загружен
    if not settings.GEMINI_API_KEY:
        print("ВНИМАНИЕ: Не установлен GEMINI_API_KEY в config/settings.py или .env")
    main()
