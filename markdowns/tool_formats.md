# Форматы определения инструментов (Function Calling)

Этот документ описывает требуемые форматы для определения инструментов (функций), которые будут использоваться с **OpenAI API** и **Google GenAI SDK** (`google-genai`).

## 1. OpenAI (Chat Completions API)

OpenAI использует структуру JSON Schema. Рекомендации для современного использования включают **Strict Mode** (Строгий режим, `"strict": true`), который гарантирует, что вывод модели будет точно соответствовать схеме.

### Структура (Strict Mode)

```json
{
  "type": "function",
  "function": {
    "name": "function_name",
    "description": "Описание функции",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {
          "type": "string",
          "description": "Описание параметра 1"
        },
        "param2": {
          "type": ["integer", "null"],
          "description": "Описание параметра 2 (необязательный)"
        }
      },
      "required": ["param1", "param2"],
      "additionalProperties": false
    }
  }
}
```

### Ключевые ограничения для Strict Mode:

1.  **`additionalProperties: false`**: Это поле **ОБЯЗАТЕЛЬНО** должно быть установлено в `false` для всех объектов в схеме.
    - **Что это значит?**: Это запрещает модели добавлять любые поля, которые не описаны в `properties`.
    - **Влияние на `**kwargs`**: Это означает, что `**kwargs` (произвольные дополнительные аргументы) **НЕ поддерживаются** в строгом режиме. Модель не сможет передать ничего, кроме того, что вы явно разрешили. Если вам нужна гибкость `**kwargs`, вам придется отказаться от `strict: true`(установить в`false`).

2.  **`required`**: Все поля, определенные в `properties`, **должны** быть перечислены в массиве `required`.
    - **Необязательные параметры**: В строгом режиме нельзя просто не включить параметр в список `required`.
    - **Как сделать параметр необязательным?**: Вы должны добавить его в `required`, но разрешить ему принимать значение `null`. В определении типа укажите `type: ["string", "null"]` (или другой тип + null). В описании (`description`) полезно подсказать модели: "Pass null if not applicable".

3.  **Поддерживаемые типы**: `string`, `number`, `integer`, `boolean`, `object`, `array`, `null`.

---

## 2. Google GenAI (Библиотека `google-genai`)

Новый Python SDK (`google-genai`) позволяет определять инструменты двумя способами: передавая обычные словари (как JSON Schema) или используя специальные типизированные объекты (`types.Schema`), что и считается "строгой типизацией" в контексте SDK.

### Вариант 1: С использованием словаря (JSON Schema)

Самый простой способ, аналогичен OpenAI, но структура немного другая.

```python
from google.genai import types

function_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="Получить погоду",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Город"},
            "days": {"type": "integer", "description": "Дни"}
        },
        "required": ["city"]
    }
)
```

### Вариант 2: С использованием строгой типизации (types.Schema)

"Строгая типизация" здесь означает использование классов `types.Schema` и перечислений `types.Type` вместо строк и словарей. Это защищает от опечаток и помогает IDE подсказывать доступные поля.

**Пример:**

```python
from google.genai import types

# Опрелеление параметров через объекты types.Schema
parameters_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "city": types.Schema(
            type=types.Type.STRING,
            description="Название города"
        ),
        "days": types.Schema(
            type=types.Type.INTEGER,
            description="Количество дней прогноза"
        ),
    },
    required=["city"]
)

# Создание декларации функции
function_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="Получить погоду для города",
    parameters=parameters_schema # Обратите внимание: используем parameters, а не parameters_json_schema
)

tool = types.Tool(
    function_declarations=[function_declaration]
)
```

### Что использование `types.Schema` дает?

1.  **Валидация на этапе написания кода**: Если вы ошибетесь в названии типа (например, напишете `itneger` вместо `INTEGER`), код упадет сразу или IDE подсветит ошибку.
2.  **Явность**: Вы используете конкретные объекты SDK, а не "магические" строки.

---

## Сравнительная таблица

| Функция                 | OpenAI (Strict)                 | Google GenAI (Dict)                     | Google GenAI (Typed)                      |
| :---------------------- | :------------------------------ | :-------------------------------------- | :---------------------------------------- |
| **Корневой объект**     | Dict с `type: "function"`       | `types.FunctionDeclaration`             | `types.FunctionDeclaration`               |
| **Определение схемы**   | `parameters` (dict)             | `parameters_json_schema` (dict)         | `parameters` (`types.Schema`)             |
| **Типы полей**          | `"string"`, `"integer"`         | `"string"`, `"integer"`                 | `types.Type.STRING`, `types.Type.INTEGER` |
| **Необязательные поля** | В `required` + тип `null`       | Не включать в `required`                | Не включать в `required`                  |
| **`**kwargs`\*\*        | **НЕТ** (запрещено strict mode) | **ДА** (если additionalProperties=True) | Зависит от схемы                          |

## Особенность обработки `null` в Python (Strict Mode)

Вы абсолютно правы. В Strict Mode модель обязана явно передать `null` для необязательного параметра.

**Проблема:**
Если у вас есть функция:

```python
def my_func(a, b="default_value"):
    print(f"a={a}, b={b}")
```

И модель пришлет JSON: `{"a": "test", "b": null}`.
При вызове `my_func(**arguments)` аргумент `b` примет значение `None`, а **не** `"default_value"`.

**Решения:**

1.  **Фильтрация в коде вызова (Middleware):**
    Перед передачей аргументов в функцию нужно удалить ключи со значением `None`, если для них в функции предусмотрены значения по умолчанию.

    ```python
    # Пример логики перед вызовом
    cleaned_args = {k: v for k, v in args.items() if v is not None}
    my_func(**cleaned_args)
    # Внимание: это сработает только если None не является валидным значением для логики функции.
    ```

2.  **Изменение сигнатуры функции:**
    Явно разрешить `None` в функции.
    ```python
    def my_func(a, b=None):
        if b is None:
            b = "default_value"
        ...
    ```

Для вашего `ToolsParser` и исполнителя инструментов **первый вариант (фильтрация)** часто является предпочтительным, чтобы не переписывать логику самих инструментов. Но нужно быть осторожным, если `None` действительно является допустимым значением, которое нужно передать.

## Обработка `*args` и `**kwargs`

Функции Python часто используют конструкцию `*args` (список позиционных аргументов) и `**kwargs` (словарь именованных аргументов). Однако **модели не понимают** этих концепций напрямую. Им нужно явное описание структуры JSON.

Вот как можно представить их в схеме:

### 1. Обработка `*args` (Список однотипных элементов)

`*args` в Python обычно используется для передачи произвольного количества аргументов. Для модели это лучше всего описывать как **один параметр типа Array**.

**Пример Python:**

```python
def sum_numbers(base, *args):
    return base + sum(args)
```

**Как описать в схеме (OpenAI & GenAI):**
Вместо `*args`, вы объявляете параметр `args` (или любое другое понятное имя, например, `numbers`), который является **массивом**.

```json
{
  "name": "sum_numbers",
  "description": "Sums the base number with a list of other numbers",
  "parameters": {
    "type": "object",
    "properties": {
      "base": { "type": "integer" },
      "numbers": {
        // Это и есть наши *args
        "type": "array",
        "items": { "type": "integer" },
        "description": "List of additional numbers to sum"
      }
    },
    "required": ["base", "numbers"],
    "additionalProperties": false
  }
}
```

**Важно:** При вызове функции в Python вам придется "распаковать" этот массив обратно в `*args`:
`func(base=args['base'], *args['numbers'])`

### 2. Обработка `**kwargs` (Произвольные именованные аргументы)

`**kwargs` позволяет передавать любые именованные параметры.

#### A. В Strict Mode (`strict: true`) - ЗАПРЕЩЕНО

В строгом режиме (OpenAI) вы **обязаны** перечислить все возможные параметры. Произвольные поля запрещены (`additionalProperties: false`).

- **Решение:** Если вы используете Strict Mode, вы **не можете** использовать `**kwargs` для "вообще любых" данных. Вы должны либо отказаться от `kwargs`, либо описать все возможные ключи, которые могут там быть, как явные необязятельные параметры.

#### B. В обычном режиме (`strict: false`) - МОЖНО

Если вы отключите `strict` режим, вы можете разрешить модели передавать дополнительные поля.

**Схема (OpenAI `strict: false` или GenAI):**
Используйте `additionalProperties` (или не указывайте его как `false`), чтобы разрешить произвольные ключи.

```json
{
  "name": "flexible_function",
  "parameters": {
    "type": "object",
    "properties": {
      "known_param": { "type": "string" }
    },
    "additionalProperties": { "type": "string" } // Разрешает любые другие ключи со строковыми значениями
    // Или просто `true` (в некоторых версиях спецификации), чтобы разрешить любой тип.
  }
}
```

**Рекомендация:**
Для надежности лучше **избегать** `**kwargs` в инструментах для LLM, так как модели часто галлюцинируют параметры, если им дать слишком много свободы. Лучше явно описывать массивы (`array`) или объекты (`object`) с конкретной структурой.

### 3. Реализация для Google GenAI с использованием `types`

В библиотеке `google-genai` для описания структуры используются объекты `types.Schema`. Вот как реализовать `*args` и гибкость (аналог `**kwargs`).

#### A. Реализация `*args` как массива

Используйте `types.Type.ARRAY` для описания аргумента, который будет принимать список значений.

```python
from google.genai import types

# Для функции def sum_numbers(base, *args):

args_schema = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(type=types.Type.INTEGER),
    description="List of additional numbers to sum"
)

parameters_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "base": types.Schema(type=types.Type.INTEGER),
        "numbers": args_schema # Представляем *args как именованный массив 'numbers'
    },
    required=["base", "numbers"]
)
```

#### B. Реализация `**kwargs` (Гибкая схема)

В `google-genai`, чтобы разрешить произвольные дополнительные поля (аналог `additionalProperties: true`), вы обычно не перечисляете их в `properties`, но сама модель Gemini по умолчанию достаточно гибкая, если не ограничена жестко.

Однако, официально в `types.Schema` не всегда есть явный аналог `additionalProperties=True` в том же виде, как в JSON Schema для всех версий SDK.

Но вы можете описать `kwargs` как **отдельный аргумент типа OBJECT**, куда модель должна положить все "extra" параметры. Это более надежный паттерн для LLM.

```python
# Для функции def config_func(name, **kwargs):
# Мы просим модель сложить все опции в один объект 'options'

options_schema = types.Schema(
    type=types.Type.OBJECT,
    description="Additional configuration options",
    # Можно не указывать properties, тогда это будет произвольный словарь
)

parameters_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "name": types.Schema(type=types.Type.STRING),
        "options": options_schema # Это и будет нашими **kwargs
    },
    required=["name"]
)
```

_Примечание_: При вызове функции вам нужно будет распаковать этот словарь: `config_func(name="...", **args['options'])`. Это самый чистый способ работы с `**kwargs` через `types`.
