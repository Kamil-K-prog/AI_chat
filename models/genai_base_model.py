from google.genai import types
from google import genai
from typing import List, Dict, Callable
from config import settings
from utils.small_utils import (
    message_helper,
    generate_timestamp,
    string_to_bytes,
    bytes_to_string,
)
import base64

from .base_model import BaseModel
import utils.types as t


class GenaiBaseModel(BaseModel):
    """Базовая модель, наследники будут только изменять что-то под конкретные типы genai моделей - уровни ризонинга, кодовое имя модели...

    Все модели принимают историю в УФС, затем конвертируют его в нужный для себя формат, делают запрос, конвертируют обратно и возвращают.
    """

    def __init__(
        self, is_reasoning=False, include_thoughts=True, reasoning_effort="medium"
    ):
        self.model_name = None  # Будет переопределено в конкретном классе модели
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.system_prompt = ""
        self.thinking_config = (
            types.ThinkingConfig(
                include_thoughts=include_thoughts, thinking_level=reasoning_effort
            )
            if is_reasoning
            else None
        )

    def _process_media_asset(self, asset: t.Asset):
        pass

    # Без обработки медиа
    def _convert_history_from_umf(self, history: t.ChatData):
        native_history = []

        for message in history.messages:
            native_parts = []
            if message.role == "system":
                self.system_prompt = message.content[0].text
            if message.role == "assistant":
                preserved_thought_signature = None
                for content in message.content:
                    if content.type == "thought":
                        preserved_thought_signature = (
                            string_to_bytes(content.signature)
                            if content.signature
                            else None
                        )
                        native_parts.append(
                            types.Part(
                                thought=True,
                                thought_signature=preserved_thought_signature,
                                text=content.text,
                            )
                        )
                    elif content.type == "text":
                        native_parts.append(types.Part(text=content.text))
                    elif content.type == "tool_call":
                        native_parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=content.tool_call.id,
                                    args=content.tool_call.args,
                                    name=content.tool_call.name,
                                ),
                                thought_signature=preserved_thought_signature,
                            )
                        )
                        preserved_thought_signature = None
                native_history.append(types.Content(role="model", parts=native_parts))

            elif message.role == "tool":
                for content in message.content:
                    native_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=content.tool_result.id,
                                name=content.tool_result.name,
                                response={
                                    "output": content.tool_result.content,
                                    "error": str(content.tool_result.is_error),
                                },
                            )
                        )
                    )
                native_history.append(
                    types.Content(
                        role="user",
                        parts=native_parts,
                    )
                )
            elif message.role == "user":
                native_history.append(
                    types.Content(
                        role="user", parts=[types.Part(text=message.content[0].text)]
                    )
                )

        return native_history

    def generate(
        self,
        history: t.ChatData,
        tools_definition,
        tools_executable: Dict[str, Callable],
    ) -> tuple[t.ChatData, list[t.Message]]:
        native_history = self._convert_history_from_umf(history)

        genai_tools = (
            [types.Tool(function_declarations=tools_definition)]
            if tools_definition
            else None
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=native_history,
            config=types.GenerateContentConfig(
                tools=genai_tools,
                system_instruction=self.system_prompt,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                thinking_config=self.thinking_config,
            ),
        )

        new_delta = []

        tool_calls = []
        content = []
        # Сначала добавляем в историю ответ модели
        for part in response.candidates[0].content.parts:
            if part.thought:
                content.append(
                    t.ThoughtContent(
                        type="thought",
                        text=part.text,
                        signature=bytes_to_string(part.thought_signature)
                        if part.thought_signature
                        else None,
                    )
                )
            elif part.text:
                content.append(t.TextContent(type="text", text=part.text))
            elif part.function_call:
                tool_call_id_fallback = message_helper.generate_id(9)
                if part.thought_signature:
                    sig_str = bytes_to_string(part.thought_signature)
                    if content and content[-1].type == "thought":
                        content[-1].signature = sig_str
                    else:
                        content.append(
                            t.ThoughtContent(
                                type="thought",
                                text="",
                                signature=sig_str,
                            )
                        )
                content.append(
                    t.ToolCallContent(
                        type="tool_call",
                        tool_call=t.ToolCall(
                            id=part.function_call.id or tool_call_id_fallback,
                            name=part.function_call.name,
                            args=part.function_call.args,
                        ),
                    )
                )
                tool_calls.append([part.function_call, tool_call_id_fallback])
        new_delta.append(
            t.Message(
                id=message_helper.generate_id(settings.MESSAGE_ID_LEN),
                role="assistant",
                content=content,
                timestamp=generate_timestamp(),
            )
        )

        # Затем цикл вызова инструментов с добавлением в историю. Добавляется один message с ролью tool, и в контенте содержатся все результаты текущего раунда вызовов
        results = []
        for tool_call, tool_call_id_fallback in tool_calls:
            is_error = False
            try:
                tool_result = tools_executable[tool_call.name](**tool_call.args)
            except Exception as e:
                is_error = True
                tool_result = str(e)
            results.append(
                t.ToolResultContent(
                    type="tool_result",
                    tool_result=t.ToolResult(
                        id=tool_call.id or tool_call_id_fallback,
                        name=tool_call.name,
                        content=str(
                            tool_result
                        ),  # Пока просто оборачиваем в str, без обработки (возможных) медиафайлов
                        is_error=is_error,
                    ),
                )
            )
        if tool_calls:
            new_delta.append(
                t.Message(
                    id=message_helper.generate_id(settings.MESSAGE_ID_LEN),
                    role="tool",
                    content=results,
                    timestamp=generate_timestamp(),
                )
            )

        # Медиафайлы и метаданные пока не обрабатываются
        history.messages.extend(new_delta)
        return history, new_delta
