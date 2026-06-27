# OpenAI Base Model
import mimetypes
import json
from openai import OpenAI
from typing import Dict, Callable
import filetype
import utils.types as t
from .base_model import BaseModel
from config import settings
from utils.small_utils import (
    message_helper,
    generate_timestamp,
    bytes_to_string,
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

    def _process_asset(self, asset: t.Asset) -> None | dict:
        # Этот метод переопределяется наследником, потому что не все модели мультимодальные
        """
        Обрабатывает ассет для загрузки в API
        Реализовать в наследниках: если ассет был загружен в облако провайдера или как-то обработан, то обновлять его в УФС, чтобы избежать повторной загрузки
        """
        pass

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
                native_content = []
                for content in message.content:
                    if self.is_thinking and content.type == "thought":
                        thought = content.text
                    elif content.type == "text":
                        native_content.append(
                            {
                                "type": "text",
                                "text": content.text
                            }
                        )
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
                    elif content.type == "media":
                        for asset in content.assets:
                            media_asset = self._process_asset(asset)
                            if media_asset:
                                native_content.append(media_asset)

                native_history.append(
                    {
                        "role": "assistant",
                        "content": native_content,
                        "reasoning_content": thought if thought else None,
                        "tool_calls": tool_calls if tool_calls else None,
                    }
                )
            elif message.role == "user":
                native_content = []
                for content in message.content:
                    if content.type == "text":
                        native_content.append(
                            {
                                "type": "text",
                                "text": content.text
                            }
                        )
                    elif content.type == "media":
                        for asset in content.assets:
                            media_asset = self._process_asset(asset)
                            if media_asset:
                                native_content.append(media_asset)
                native_history.append(
                    {
                        "role": "user",
                        "content": native_content
                    }
                )


            elif message.role == "tool":
                for content in message.content:
                    native_history.append(
                        {
                            "role": "tool",
                            "content": f"Function response: ```{content.tool_result.text_content}```\nError: {content.tool_result.is_error}",
                            "tool_call_id": content.tool_result.id,
                        }
                    )
        return native_history

    def _do_request(self, native_history, tools_definition, extra_body=None):
        if extra_body is None:
            extra_body = {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=native_history,
            tools=tools_definition,
            extra_body=extra_body
        )
        return response

    def generate(
            self,
            history: t.ChatData,
            tools_definition,
            tools_executable: Dict[str, Callable],
            extra_body: dict
    ) -> tuple[t.ChatData, list[t.Message]]:
        native_history = self._convert_history_from_umf(history)

        response = self._do_request(native_history, tools_definition, extra_body)
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
                            id=tool_call.id or tool_call_id_fallback,
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
            tool_result_asset = None
            try:
                current_tool = tools_executable[tool_call.function.name]
                # Если функция возвращает медиа
                if getattr(current_tool, "returns_media", False):
                    tool_result = tools_executable[tool_call.function.name](**json.loads(tool_call.function.arguments))

                    # Распаковка результатов функции
                    tool_result_str = "Медиафайл успешно сгенерирован"
                    media_bytes = b""
                    mime_type_str = None
                    if isinstance(tool_result, tuple):
                        if len(tool_result) == 3:
                            tool_result_str, media_bytes, mime_type_str = tool_result
                        elif len(tool_result) == 2:
                            # Предположим, вернули (bytes, mime_type) или (text, bytes)
                            if isinstance(tool_result[0], bytes):
                                media_bytes, mime_type_str = tool_result
                            else:
                                tool_result_str, media_bytes = tool_result
                    elif isinstance(tool_result, bytes):
                        media_bytes = tool_result

                    asset_id = message_helper.generate_id(settings.ASSET_ID_LEN)

                    # Угадываем MIME тип
                    kind = filetype.guess(media_bytes)
                    mime_type = getattr(current_tool, "mime_type", None) or mime_type_str or (
                        kind.mime if kind else "application/octet-stream")
                    if mime_type.startswith("image/"):
                        asset_type = "image"
                    elif mime_type.startswith("video/"):
                        asset_type = "video"
                    elif mime_type.startswith("audio/"):
                        asset_type = "audio"
                    else:
                        asset_type = "document"

                    # Сохраняем медиа файл
                    ext = mimetypes.guess_extension(mime_type) or ".bin"
                    asset_local_path = f"{settings.MEDIA_FOLDER}/{asset_id}{ext}"
                    with open(asset_local_path, "wb") as f:
                        f.write(media_bytes)

                    # Добавляем ассет
                    tool_result_asset = [t.Asset(
                        id=asset_id,
                        type=asset_type,
                        local_path=asset_local_path,
                        mime_type=mime_type,
                        size_bytes=len(media_bytes),
                        data_base64=bytes_to_string(media_bytes) if len(media_bytes) < 20 * 1024 * 1024 else None
                    )]
                else:
                    tool_result_str = tools_executable[tool_call.function.name](**json.loads(tool_call.function.arguments))

            except Exception as e:
                is_error = True
                tool_result_str = str(e)

            results.append(
                t.ToolResultContent(
                    type="tool_result",
                    tool_result=t.ToolResult(
                        id=tool_call.id or tool_call_id_fallback,
                        name=tool_call.function.name,
                        text_content=str(tool_result_str),
                        is_error=is_error,
                    ),
                    assets=tool_result_asset
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
