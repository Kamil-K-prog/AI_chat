# OpenAI Base Model
from openai import OpenAI
from typing import Dict, Callable
import json
import utils.types as t
from .base_model import BaseModel
from config import settings
from utils.small_utils import (
    message_helper,
    generate_timestamp,
)


class OpenAiBaseModel(BaseModel):
    """Базовая модель, наследники будут только изменять что-то под конкретные типы openai моделей - уровни ризонинга, кодовое имя модели...
    Все модели принимают историю в УФС, затем конвертируют его в нужный для себя формат, делают запрос, конвертируют обратно и возвращают.
    """

    def __init__(
            self,
            model_name,
            system_prompt="",
            base_url=None,
            api_key=None,
            is_thinking=False,
    ):
        self.model_name = model_name
        self.client = self._create_client(base_url, api_key)
        self.system_prompt = system_prompt
        self.is_thinking = is_thinking

    def _create_client(self, base_url, api_key) -> OpenAI:
        client = OpenAI(api_key=api_key)
        if base_url:
            client.base_url = base_url
        return client

    def _process_media_asset(self):
        pass

    # Без обработки медиа
    def _convert_history_from_umf(self, history: t.ChatData):
        native_history = []

        for message in history.messages:
            if message.role == "system":
                native_history.append(
                    {"role": "system", "content": message.content[0].text}
                )
                self.system_prompt = message.content[0].text
            elif message.role == "assistant":
                thought = ""
                tool_calls = []
                text = ""
                for content in message.content:
                    if self.is_thinking and content.type == "thought":
                        thought = content.text
                    elif content.type == "text":
                        text += content.text
                    elif content.type == "tool_call":
                        tool_calls.append(
                            {
                                "id": content.tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": content.tool_call.name,
                                    "arguments": json.dumps(content.tool_call.args),
                                },
                            }
                        )
                native_history.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "reasoning_content": thought if thought else None,
                        "tool_calls": tool_calls if tool_calls else None,
                    }
                )
            elif message.role == "user":
                native_history.append(
                    {"role": "user", "content": message.content[0].text}
                )
            elif message.role == "tool":
                for content in message.content:
                    native_history.append(
                        {
                            "role": "tool",
                            "content": f"Function response: ```{content.tool_result.content}```\nError: {content.tool_result.is_error}",
                            "tool_call_id": content.tool_result.id,
                        }
                    )
        return native_history

    def _do_request(self, native_history, tools_definition):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=native_history,
            tools=tools_definition,
        )
        return response

    def generate(
            self,
            history: t.ChatData,
            tools_definition,
            tools_executable: Dict[str, Callable],
    ) -> tuple[t.ChatData, list[t.Message]]:
        native_history = self._convert_history_from_umf(history)
        # print(f"Отладка: {native_history}")

        response = self._do_request(native_history, tools_definition)
        message = response.choices[0].message

        new_delta = []
        tool_calls = []

        content = []
        if self.is_thinking and message.reasoning_content:
            content.append(t.ThoughtContent(type="thought", text=message.reasoning_content))
        if message.content:
            content.append(t.TextContent(type="text", text=message.content or ""))
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_id_fallback = message_helper.generate_id(9)
                tool_calls.append([tool_call, tool_call_id_fallback])
                content.append(
                    t.ToolCallContent(
                        type="tool_call",
                        tool_call=t.ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            args=json.loads(tool_call.function.arguments)
                        )
                    )
                )
        # Добавляем сообщение ассистента в историю
        new_delta.append(t.Message(
            id=message_helper.generate_id(settings.MESSAGE_ID_LEN),
            role="assistant",
            content=content,
            timestamp=generate_timestamp()
        ))

        # Выполняем функции и добавляем их в историю
        results = []
        for tool_call, tool_call_id_fallback in tool_calls:
            is_error = False
            try:
                tool_result = tools_executable[tool_call.function.name](**json.loads(tool_call.function.arguments))
            except Exception as e:
                is_error = True
                tool_result = str(e)
            results.append(
                t.ToolResultContent(
                    type="tool_result",
                    tool_result=t.ToolResult(
                        id=tool_call.id or tool_call_id_fallback,
                        name=tool_call.function.name,
                        content=str(
                            tool_result
                        ),  # Пока просто оборачиваем в str, без обработки медиафайлов
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

        history.messages.extend(new_delta)
        return history, new_delta
