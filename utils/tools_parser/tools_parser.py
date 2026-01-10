from inspect import signature, Parameter
from docstring_parser import parse
import typing

type_mapping = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolsParser:
    _registry = []

    @property
    def tools(self):
        return self._registry

    @classmethod
    def register_tool(cls, func):
        cls._registry.append(func)

    def get_tools_dict(self) -> dict:
        """Возвращает словарь инструментов, чтобы вызывать их в моделях"""
        return {tool.__name__: tool for tool in self._registry}

    def _find_arg_index(self, param_object, parsed_docstring_params) -> int:
        for i, param in enumerate(parsed_docstring_params):
            if param.arg_name == param_object.name:
                return i
        return -1

    def _get_type(self, annotation):
        if annotation == Parameter.empty:
            res = "string"
        elif not isinstance(annotation, type) and not typing.get_origin(annotation):
            res = type_mapping.get(type(annotation), "string")
        else:
            origin = typing.get_origin(annotation) or annotation
            res = type_mapping.get(origin, "string")

        return res

    def _get_tool_json(self, tool):
        """Возвращает JSON-схему конкретного инструмента в базовом формате."""
        parsed_docstring = parse(tool.__doc__)
        sig = signature(tool)

        required_params = []
        properties_json = {}

        for param_name, param_object in sig.parameters.items():
            # Определяем тип: приоритет у аннотации, если её нет — берем тип значения по умолчанию
            annotation = param_object.annotation
            if (
                annotation == Parameter.empty
                and param_object.default != Parameter.empty
            ):
                annotation = type(param_object.default)

            json_type = self._get_type(annotation)

            # Ищем описание в docstring
            param_index = self._find_arg_index(param_object, parsed_docstring.params)
            param_description = (
                parsed_docstring.params[param_index].description
                if param_index != -1
                else "No description"
            )

            properties_json[param_name] = {
                "type": json_type,
                "description": param_description,
            }

            # Для strict mode все параметры должны быть обязательными
            required_params.append(param_name)

        func_description = f"{parsed_docstring.short_description}\n{parsed_docstring.long_description or ''}".strip()
        return {
            "name": tool.__name__,
            "description": func_description,
            "parameters": {
                "type": "object",
                "properties": properties_json,
                "required": required_params,
                "additionalProperties": False,
            },
        }

    def get_tools_json_openai(self):
        return [
            {
                "type": "function",
                "function": {**self._get_tool_json(t), "strict": True},
            }
            for t in self.tools
        ]

    def get_tools_scheme_genai(self):
        from google.genai import types

        functions = []
        for tool in self.tools:
            tool_json = self._get_tool_json(tool)
            functions.append(
                types.FunctionDeclaration(
                    name=tool_json["name"],
                    description=tool_json["description"],
                    parameters_json_schema={
                        "type": "object",
                        "properties": tool_json["parameters"]["properties"],
                        "required": tool_json["parameters"]["required"],
                    },
                )
            )
        return types.Tool(function_declarations=functions)


# Декоратор для регистрации инструментов. Добавляет в пул сами объекты
def register_tool(func):
    ToolsParser.register_tool(func)
    return func
