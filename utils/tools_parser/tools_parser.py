from inspect import signature, Parameter
from typing import get_origin, get_args
from docstring_parser import parse
from google.genai import types

"""
В этом файле находится парсер функций, который превращает их в описание для моделей openai и genai. 

Важное замечание: если функция содержит *args или **kwargs, то используйте эти общепринятые названия. 
В других частях программы (при выполнении tool_calls) эти названия зарезервированы и используются парсером аргументов.
"""

class ToolsParser:
    _registry = []

    @classmethod
    def get_tools(cls) -> list:
        """Возвращает список объектов инструментов"""
        return cls._registry

    @classmethod
    def register_tool(cls, func):
        cls._registry.append(func)

    @classmethod
    def get_tools_callables(cls):
        """Возвращает словарь функций, которые можно вызвать"""
        return {tool.__name__: tool for tool in cls.get_tools()}

    @classmethod
    def _get_annotation_schema_openai(cls, annotation):
        """Рекурсивно преобразует аннотацию типа в часть JSON-схемы"""
        if annotation == Parameter.empty:
            return {"type": "string"}

        origin = get_origin(annotation)
        args = get_args(annotation)

        types_mapping_openai = {
            int: "integer",
            float: "number",
            bool: "boolean",
            str: "string",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        # Если это простой тип (не Generic)
        if origin is None:
            # Проверяем, есть ли тип в маппинге
            schema_type = types_mapping_openai.get(annotation, "string")
            # Если аннотация list или dict, но без уточнения типов (просто list вместо list[int])
            return {"type": schema_type}

        # Если это Generic
        if origin is list:
            item_schema = cls._get_annotation_schema_openai(args[0]) if args else {}
            return {"type": "array", "items": item_schema}

        # Обработка других generics при необходимости
        return {"type": "object"}

    @classmethod
    def _get_annotation_type_genai(cls, annotation):
        """Рекурсивно преобразует аннотацию типа в types.Type"""
        if annotation == Parameter.empty:
            return types.Type.STRING

        origin = get_origin(annotation)
        args = get_args(annotation)

        types_mapping_genai = {
            int: types.Type.INTEGER,
            float: types.Type.NUMBER,
            bool: types.Type.BOOLEAN,
            str: types.Type.STRING,
            list: types.Type.ARRAY,
            dict: types.Type.OBJECT,
            type(None): types.Type.NULL,
        }

        if origin is None:
            return types_mapping_genai.get(annotation, types.Type.STRING)

        if origin is list:
            item_type = (
                cls._get_annotation_type_genai(args[0]) if args else types.Type.STRING
            )
            return item_type

        return types.Type.OBJECT

    @classmethod
    def get_json_schema_openai(cls, strict_mode=True, ignore_kwarg_funcs=False):
        """Возвращает JSON Schema для openai-совместимых моделей

        additionalProperties: false — это правило: "Нельзя иметь лишние поля"
        strict: true — это режим принуждения: "Модель, следуй этому правилу (и всем остальным) со 100% гарантией".

        Автоматически проверяет наличие **kwargs, и устанавливает additionalProperties
        Если strict=True, но функция содержит **kwargs, то:
            1) При ignore_kwarg_funcs=False выбрасывает ошибку
            2) При ignore_kwarg_funcs=True не добавляет функцию в схему вовсе

        *args воспринимается как переменная `args` типа list, и передаётся даже в strict mode

        Сейчас значение strict_mode едино для всех функций, хотя его передают с каждой функцией отдельно.
        В strict_mode, если параметр необязательный (имеет значение по умолчанию), он обозначается как обязательный, но с возможностью использовать значение по умолчанию (если модель вернёт 'null' в качестве значения)
        """
        res = []
        for tool in cls.get_tools():
            ignore_this_function = False
            function_json = {}
            docstring = parse(tool.__doc__)
            sig = signature(tool)
            additional_properties = False

            # Заполняем JSON параметров
            function_properties = {}
            required_properties = []
            for index, (name, param) in enumerate(sig.parameters.items()):
                try:
                    param_description = docstring.params[index].description
                except IndexError:
                    param_description = "No description."

                if param.default != Parameter.empty:
                    param_description += (
                        f" (System: Optional. Default value: {param.default}"
                    )
                    if strict_mode:  # Отдельная обработка для strict_mode
                        param_description += (
                            " Pass null if you want to use default value."
                        )
                    param_description += ")."
                if param.default == Parameter.empty or strict_mode:
                    required_properties.append(name)

                if (
                    param.kind == Parameter.POSITIONAL_OR_KEYWORD
                ):  # Обрабатываем обычные аргументы
                    schema_part = cls._get_annotation_schema_openai(param.annotation)

                    if param.default != Parameter.empty and strict_mode:
                        # Если strict_mode и есть дефолтное значение, разрешаем null
                        current_type = schema_part.get("type")
                        if isinstance(current_type, list):
                            if "null" not in current_type:
                                schema_part["type"] = current_type + ["null"]
                        else:
                            schema_part["type"] = [current_type, "null"]

                    schema_part["description"] = param_description
                    function_properties[name] = schema_part

                elif param.kind == Parameter.VAR_POSITIONAL:  # Обрабатываем *args
                    # args - это всегда список, элементы которого имеют тип param.annotation
                    # Например, если *args: int, то args это [int, int, ...]
                    # Если *args: list[int], то args это [[int], [int], ...]

                    item_schema = cls._get_annotation_schema_openai(param.annotation)

                    function_properties[name] = {
                        "type": "array",
                        "items": item_schema,
                        "description": param_description
                        + " (System: this variable is an array of positional arguments (*args)).",
                    }
                elif param.kind == Parameter.VAR_KEYWORD:  # Обрабатываем **kwargs
                    if strict_mode:
                        if not ignore_kwarg_funcs:
                            raise ValueError(
                                f"Функция {tool.__name__} имеет параметр **kwargs, который не разрешен в строгом режиме"
                            )
                        else:
                            ignore_this_function = True
                            break
                    additional_properties = True
                    # Не добавляем здесь никакого объекта, в который будут класться **kwargs, модель просто предоставит их в генерации

            if ignore_this_function:
                continue

            function_json["name"] = tool.__name__
            function_json["description"] = docstring.description
            function_json["strict"] = strict_mode
            function_json["parameters"] = {
                "type": "object",
                "properties": function_properties,
                "required": required_properties,
                "additionalProperties": additional_properties,
            }
            res.append({"type": "function", "function": function_json})
        return res

    @classmethod
    def get_types_schema_genai(cls):
        """Собирает типизированную схему представления инструментов для genai-совместимых моделей"""
        res = []

        for tool in cls.get_tools():
            docstring = parse(tool.__doc__)
            sig = signature(tool)

            function_properties = {}
            required_properties = []
            for index, (name, param) in enumerate(sig.parameters.items()):
                try:
                    param_description = docstring.params[index].description
                except IndexError:
                    param_description = "No description."

                if param.default == Parameter.empty:
                    required_properties.append(name)

                param_type = cls._get_annotation_type_genai(
                    param.annotation
                )  # Либо тип самой переменной, либо тип переменных, содержащихся в массиве (Максимальная поддерживаемая вложенность list[int], более глубокая вложенность list[list[int]] будет сведена к list[int])

                param_schema = types.Schema(
                    description=param_description, type=param_type
                )

                if param.default != Parameter.empty:
                    param_schema.default = param.default

                if get_origin(param.annotation) is list:
                    param_schema.type = types.Type.ARRAY
                    param_schema.items = types.Schema(type=param_type)

                if param.kind == Parameter.VAR_POSITIONAL:
                    param_schema.description += " (System: this variable is an array of positional arguments (*args))."
                    param_schema.type = types.Type.ARRAY
                    param_schema.items = types.Schema(type=param_type)
                if param.kind == Parameter.VAR_KEYWORD:
                    param_schema.description += " (System: this variable is a dictionary of keyword arguments (**kwargs))."
                    param_schema.type = types.Type.OBJECT

                function_properties[name] = param_schema

            # Собираем вместе
            parameters_schema = types.Schema(
                type=types.Type.OBJECT,
                properties=function_properties,
                required=required_properties,
            )
            res.append(
                types.FunctionDeclaration(
                    name=tool.__name__,
                    description=docstring.description,
                    parameters=parameters_schema,
                )
            )

        return res


# Декоратор для регистрации инструментов. Добавляет в пул сами объекты
def register_tool(func):
    ToolsParser.register_tool(func)
    return func
