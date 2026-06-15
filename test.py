from models.genai.gemini import Gemini3_1FlashLite
from models.openai.deepseek import DeepseekChat, DeepseekReasoner
from models import BaseModel
import utils.types as t
from utils.small_utils import generate_timestamp
from utils.tools_parser.tools_parser import ToolsParser
import utils.tools # noqa: F401


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
                        text="Сложи два числа 5634 и 77854.1 с помощью bar_func"
                    )
                ],
            ),
        ],
    )


def auto_test(model: BaseModel, tools_def, tools_exec):
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
    # print("Отладка: \n", history.model_dump_json(indent=2))
    for msg in delta2:
        print(f"Роль: {msg.role}")
        for c in msg.content:
            if c.type == "text":
                print(f"  Текст: {c.text}")
            elif c.type == "thought":
                print(f"  Мысли: {c.text}")


def chat_mode(model: BaseModel, tools_def, tools_exec, provider: str = "genai"):
    from utils.small_utils.messages_helper import message_helper

    print("\n--- Режим интерактивного чата (для выхода введите 'exit' или 'выход') ---")

    # Инициализируем пустую историю
    history = t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(provider=provider)), messages=[]
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
def test_model_switching():
    print("\n--- Запуск теста переключения моделей на лету (GenAI <-> OpenAI) ---")
    
    # 1. Создаем модели
    print("Инициализация моделей...")
    gemini_model = Gemini3_1FlashLite(
        system_prompt="Ты полезный ИИ-ассистент. Отвечай очень кратко, буквально одним-двумя словами."
    )
    deepseek_model = DeepseekChat(
        system_prompt="Ты полезный ИИ-ассистент. Отвечай очень кратко, буквально одним-двумя словами."
    )
    
    # 2. Инициализируем историю
    history = t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(provider="genai")),
        messages=[]
    )
    
    # --- ШАГ 1: Запрос к Gemini (GenAI) ---
    print("\n[Шаг 1] Запрос к Gemini (GenAI)...")
    history.messages.append(
        t.Message(
            id="msg_user_1",
            timestamp=generate_timestamp(),
            role="user",
            content=[t.TextContent(text="Привет! Назови одно случайное секретное слово (существительное).")]
        )
    )
    
    # У Gemini нет инструментов в этом тесте
    history, delta1 = gemini_model.generate(history, tools_definition=[], tools_executable={})
    
    gemini_word = ""
    for msg in delta1:
        if msg.role == "assistant":
            for content in msg.content:
                if content.type == "text":
                    print(f"  [Gemini]: {content.text}")
                    gemini_word += content.text
                    
    # --- ШАГ 2: Переключение на DeepSeek (OpenAI) ---
    print("\n[Шаг 2] Переключение на DeepSeek (OpenAI)...")
    history.chat_metadata.config.provider = "openai"
    
    history.messages.append(
        t.Message(
            id="msg_user_2",
            timestamp=generate_timestamp(),
            role="user",
            content=[t.TextContent(text="Повтори секретное слово, которое ты назвал в предыдущем сообщении. Ответь только этим словом.")]
        )
    )
    
    history, delta2 = deepseek_model.generate(history, tools_definition=[], tools_executable={})
    
    deepseek_word = ""
    for msg in delta2:
        if msg.role == "assistant":
            for content in msg.content:
                if content.type == "text":
                    print(f"  [DeepSeek]: {content.text}")
                    deepseek_word += content.text
                    
    # --- ШАГ 3: Переключение обратно на Gemini (GenAI) ---
    print("\n[Шаг 3] Переключение обратно на Gemini (GenAI)...")
    history.chat_metadata.config.provider = "genai"
    
    history.messages.append(
        t.Message(
            id="msg_user_3",
            timestamp=generate_timestamp(),
            role="user",
            content=[t.TextContent(text="Какое слово мы назвали в самом первом сообщении? Ответь только им.")]
        )
    )
    
    history, delta3 = gemini_model.generate(history, tools_definition=[], tools_executable={})
    
    final_word = ""
    for msg in delta3:
        if msg.role == "assistant":
            for content in msg.content:
                if content.type == "text":
                    print(f"  [Gemini (снова)]: {content.text}")
                    final_word += content.text
                    
    print("\n--- Тест завершен ---")
    print(f"Gemini назвал: '{gemini_word.strip()}'")
    print(f"DeepSeek повторил: '{deepseek_word.strip()}'")
    print(f"Gemini в конце подтвердил: '{final_word.strip()}'")

def test_react_switching():
    print("\n--- Запуск теста ReAct и переключения моделей на лету (GenAI <-> OpenAI) ---")
    
    # 1. Загружаем определения инструментов
    tools_exec = ToolsParser.get_tools_callables()
    tools_def_genai = ToolsParser.get_types_schema_genai()
    tools_def_openai = ToolsParser.get_json_schema_openai()
    
    # 2. Создаем модели
    print("Инициализация моделей...")
    gemini_model = Gemini3_1FlashLite(
        system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко."
    )
    deepseek_model = DeepseekChat(
        system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # СЦЕНАРИЙ А: Успешная смена модели ПОСЛЕ завершения цикла ReAct
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- [Сценарий А] Успешная смена модели ПОСЛЕ завершения цикла ReAct ---")
    history_a = t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(provider="openai")),
        messages=[]
    )
    history_a.messages.append(
        t.Message(
            id="msg_user_react_a",
            timestamp=generate_timestamp(),
            role="user",
            content=[t.TextContent(text="Узнай погоду за окном у пользователя, затем с помощью функции прибавь к результату 25 и выведи ответ")]
        )
    )
    
    print("Запуск цикла ReAct для DeepSeek...")
    while True:
        history_a, delta = deepseek_model.generate(
            history=history_a,
            tools_definition=tools_def_openai,
            tools_executable=tools_exec
        )
        
        for msg in delta:
            print(f"  Роль: {msg.role}")
            for c in msg.content:
                if c.type == "tool_call":
                    print(f"    Вызов инструмента: {c.tool_call.name}({c.tool_call.args})")
                elif c.type == "tool_result":
                    print(f"    Результат: {c.tool_result.content}")
                elif c.type == "text":
                    print(f"    Текст: {c.text}")
                    
        if delta[-1].role == "tool":
            continue
        else:
            break
            
    # Переключаемся на Gemini после завершения цикла
    print("\nПереключаемся на Gemini после полного ReAct цикла...")
    history_a.chat_metadata.config.provider = "genai"
    history_a.messages.append(
        t.Message(
            id="msg_user_react_a_followup",
            timestamp=generate_timestamp(),
            role="user",
            content=[t.TextContent(text="Какая погода была получена и какое число к ней прибавили? Ответь кратко.")]
        )
    )
    
    history_a, delta_gemini = gemini_model.generate(
        history=history_a,
        tools_definition=tools_def_genai,
        tools_executable=tools_exec
    )
    
    for msg in delta_gemini:
        if msg.role == "assistant":
            for c in msg.content:
                if c.type == "text":
                    print(f"  [Gemini]: {c.text}")

    # ══════════════════════════════════════════════════════════════════════════
    # СЦЕНАРИЙ Б: Ошибка 400 при смене модели ПОСЕРЕДИНЕ цикла ReAct
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- [Сценарий Б] Ошибка 400 при смене модели ПОСЕРЕДИНЕ цикла ReAct ---")
    history_b = t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(provider="openai")),
        messages=[]
    )
    history_b.messages.append(
        t.Message(
            id="msg_user_react_b",
            timestamp=generate_timestamp(),
            role="user",
            content=[t.TextContent(text="Узнай погоду за окном у пользователя, затем с помощью функции прибавь к результату 25 и выведи ответ")]
        )
    )
    
    # 1. Первый шаг DeepSeek. Мы ожидаем, что он вызовет current_weather.
    print("Вызов DeepSeek (шаг 1)...")
    history_b, delta1 = deepseek_model.generate(
        history=history_b,
        tools_definition=tools_def_openai,
        tools_executable=tools_exec
    )
    
    has_tool_call = False
    for msg in delta1:
        print(f"  Роль: {msg.role}")
        for c in msg.content:
            if c.type == "tool_call":
                print(f"    Вызов инструмента: {c.tool_call.name}({c.tool_call.args})")
                has_tool_call = True
            elif c.type == "tool_result":
                print(f"    Результат: {c.tool_result.content}")
                
    if not has_tool_call:
        print("  [Предупреждение] Модель не вызвала инструмент на первом шаге!")
        
    # 2. Переключаемся на Gemini посередине цикла ReAct
    print("\nПереключаемся на Gemini... (ожидаем ошибку 400)")
    history_b.chat_metadata.config.provider = "genai"
    
    from google.genai import errors
    try:
        history_b, delta_gemini = gemini_model.generate(
            history=history_b,
            tools_definition=tools_def_genai,
            tools_executable=tools_exec
        )
        print("  [ОШИБКА ТЕСТА] Gemini успешно ответила без ошибки 400!")
    except errors.APIError as e:
        print(f"  [УСПЕХ] Перехвачена ожидаемая ошибка от Gemini API (код {e.code}):")
        print(f"  Детали ошибки: {e.message}")
    except Exception as e:
        print(f"  [УСПЕХ] Перехвачена ошибка: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # СЦЕНАРИЙ В: Успешная смена модели ПОСЕРЕДИНЕ цикла ReAct (GenAI -> OpenAI)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- [Сценарий В] Успешная смена модели ПОСЕРЕДИНЕ цикла ReAct (GenAI -> OpenAI) ---")
    history_c = t.ChatData(
        chat_metadata=t.ChatMetadata(config=t.ChatConfig(provider="genai")),
        messages=[]
    )
    history_c.messages.append(
        t.Message(
            id="msg_user_react_c",
            timestamp=generate_timestamp(),
            role="user",
            content=[t.TextContent(text="Узнай погоду за окном у пользователя, затем с помощью функции прибавь к результату 25 и выведи ответ")]
        )
    )
    
    # 1. Первый шаг Gemini. Мы ожидаем, что она вызовет current_weather.
    print("Вызов Gemini (шаг 1)...")
    history_c, delta_c1 = gemini_model.generate(
        history=history_c,
        tools_definition=tools_def_genai,
        tools_executable=tools_exec
    )
    
    has_tool_call_c = False
    for msg in delta_c1:
        print(f"  Роль: {msg.role}")
        for c in msg.content:
            if c.type == "tool_call":
                print(f"    Вызов инструмента: {c.tool_call.name}({c.tool_call.args})")
                has_tool_call_c = True
            elif c.type == "tool_result":
                print(f"    Результат: {c.tool_result.content}")
                
    if not has_tool_call_c:
        print("  [Предупреждение] Gemini не вызвала инструмент на первом шаге!")

    # 2. Переключаемся на DeepSeek посередине цикла ReAct
    print("\nПереключаемся на DeepSeek... (ожидаем продолжение ReAct без ошибок)")
    history_c.chat_metadata.config.provider = "openai"
    
    # Запускаем цикл DeepSeek до финального ответа
    while True:
        history_c, delta_c2 = deepseek_model.generate(
            history=history_c,
            tools_definition=tools_def_openai,
            tools_executable=tools_exec
        )
        
        for msg in delta_c2:
            print(f"  Роль: {msg.role}")
            for c in msg.content:
                if c.type == "tool_call":
                    print(f"    Вызов инструмента: {c.tool_call.name}({c.tool_call.args})")
                elif c.type == "tool_result":
                    print(f"    Результат: {c.tool_result.content}")
                elif c.type == "text":
                    print(f"    Текст: {c.text}")
                    
        if delta_c2[-1].role == "tool":
            continue
        else:
            break

    print("\n--- Тест ReAct завершен ---")


def main():
    tools_exec = ToolsParser.get_tools_callables()
    print(f"Доступные инструменты: {list(tools_exec.keys())}")

    # Выбор модели
    print("\nВыберите модель:")
    print("1 - Gemini 3.1 Flash Lite")
    print("2 - DeepSeek Chat")
    print("3 - DeepSeek Reasoner")
    model_choice = input("> ").strip()

    if model_choice == "1":
        model = Gemini3_1FlashLite(
            system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко."
        )
        tools_def = ToolsParser.get_types_schema_genai()
        provider = "genai"
    elif model_choice == "2":
        model = DeepseekChat(
            system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко."
        )
        tools_def = ToolsParser.get_json_schema_openai()
        provider = "openai"
    elif model_choice == "3":
        model = DeepseekReasoner(
            system_prompt="Ты полезный ИИ-ассистент. Отвечай кратко."
        )
        tools_def = ToolsParser.get_json_schema_openai()
        provider = "openai"
    else:
        print("Неверный выбор.")
        return

    print(f"Модель: {model.model_name}")

    while True:
        print("\nВыберите режим:")
        print("1 - Автоматическая проверка (старый тест)")
        print("2 - Интерактивный чат")
        print("3 - Тест смены моделей на лету (GenAI <-> OpenAI)")
        print("4 - Тест ReAct и подмены моделей посередине цикла (GenAI <-> OpenAI)")
        print("0 - Выход")
        choice = input("> ").strip()

        if choice == "1":
            auto_test(model, tools_def, tools_exec)
        elif choice == "2":
            chat_mode(model, tools_def, tools_exec, provider)
        elif choice == "3":
            test_model_switching()
        elif choice == "4":
            test_react_switching()
        elif choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте еще раз.")


if __name__ == "__main__":
    main()
