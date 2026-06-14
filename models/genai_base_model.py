import base64
from statistics import median

from google.genai import types, errors
from google import genai
from typing import Dict, Callable, List
from config import settings
from utils.small_utils import (
    message_helper,
    generate_timestamp,
    string_to_bytes,
    bytes_to_string,
    file_to_bytes
)
from utils.types import Asset
from .base_model import BaseModel
import utils.types as t


class GenaiBaseModel(BaseModel):
    """Базовая модель, наследники будут только изменять что-то под конкретные типы genai моделей - уровни ризонинга, кодовое имя модели...

    Все модели принимают историю в УФС, затем конвертируют его в нужный для себя формат, делают запрос, конвертируют обратно и возвращают.
    """

    def __init__(
            self, model_name, is_reasoning=False, include_thoughts=True, reasoning_effort="medium", system_prompt=""
    ):
        self.model_name = model_name
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.system_prompt = system_prompt
        self.thinking_config = (
            types.ThinkingConfig(
                include_thoughts=include_thoughts, thinking_level=reasoning_effort
            )
            if is_reasoning
            else None
        )

    def _process_asset(self, asset: t.Asset) -> t.Asset:
        """
        Полный цикл работы с ассетом (медиафайлом): если он есть в облаке, то добавляем файл и возвращаем, если нет, то загружаем и возвращаем
        Файл больше 20 МБ, иначе кодируем в Base64
        """
        filename = asset.cloud_refs.genai.filename
        try:
            # Если файл уже есть в облаке, то просто кладём его в объект, остальные метаданные верные
            file = self.client.files.get(filename)
            asset.cloud_refs.genai.file_object = file
            return asset
        except errors.APIError as e:
            if e.status_code == 404:
                # Если файл не найден, то нужно загрузить его и установить все нужные поля
                file = self.client.files.upload(file=asset.local_path)
                asset.cloud_refs.genai.filename = file.filename
                asset.cloud_refs.genai.file_object = file
                asset.cloud_refs.genai.uri = file.uri
            return asset

    def _convert_history_from_umf(self, history: t.ChatData) -> List[types.Content]:
        """
        Конвертирует из УФС в нативный для genai формат
        :param history:
        :return:
        """
        native_history = []

        for message in history.messages:
            native_parts = []
            if message.role == "system":
                self.system_prompt = message.content[0].text
            if message.role == "assistant":
                preserved_thought_signature = None
                for content in message.content:
                    if content.type == "thought":
                        if content.signature:  # Если ответ от модели genai, то есть подпись, и эту CoT можно подать на вход. Если мысли не подписаны, то API вернет ошибку
                            # --- ПРОБЛЕМА --- Начиная с Gemini 3 если не вернуть мысли в цикле ReAct, то API вернёт ошибку 400 https://ai.google.dev/gemini-api/docs/thought-signatures?hl=ru#model-behavior
                            preserved_thought_signature = string_to_bytes(content.signature)
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
                        # Забыл, зачем эта строка:preserved_thought_signature = None
                    elif content.type == "media":
                        for asset in content.assets:
                            if asset.size_bytes < 20 * 1024 * 1024:  # Файлы меньше 20 Мб посылаем как строки
                                if asset.data_base64:
                                    raw_bytes = string_to_bytes(asset.data_base64)
                                else:
                                    raw_bytes = file_to_bytes(asset.local_path)
                                media_part = types.Part(
                                    inline_data=types.Blob(
                                        data=raw_bytes,
                                        mime_type=asset.mime_type
                                    )
                                )
                            else:  # Файлы больше 20 МБ добавляем как URI, ведущий на google cloud
                                media_asset = self._process_asset(asset)
                                media_part = types.Part(
                                    file_data=types.FileData(
                                        file_uri=media_asset.cloud_refs.genai.uri,
                                        mime_type=asset.mime_type
                                    )
                                )
                            native_parts.append(media_part)

                native_history.append(types.Content(role="model", parts=native_parts))

            elif message.role == "tool":
                for content in message.content:
                    media_parts = []
                    tool_part = types.Part(
                        function_response=types.FunctionResponse(
                            id=content.tool_result.id,
                            name=content.tool_result.name,
                            response={
                                "output": content.tool_result.content,
                                "error": str(content.tool_result.is_error),
                            },
                        )
                    )
                    for asset in content.assets:
                        if asset.size_bytes < 20 * 1024 * 1024:  # Файлы меньше 20 Мб посылаем как строки
                            if asset.data_base64:
                                raw_bytes = string_to_bytes(asset.data_base64)
                            else:
                                raw_bytes = file_to_bytes(asset.local_path)
                            function_response_part = types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    data=raw_bytes,
                                    mime_type=asset.mime_type
                                )
                            )
                        else:  # Файлы больше 20 МБ добавляем как URI, ведущий на google cloud
                            media_asset = self._process_asset(asset)
                            function_response_part = types.FunctionResponsePart(
                                file_data=types.FunctionResponseFileData(
                                    file_uri=media_asset.cloud_refs.genai.uri,
                                    mime_type=asset.mime_type
                                )
                            )
                        media_parts.append(function_response_part)
                    tool_part.function_response.parts = media_parts
                    native_parts.append(tool_part)

                native_history.append(
                    types.Content(
                        role="user",
                        parts=native_parts,
                    )
                )

            elif message.role == "user":
                for content in message.content:
                    if content.type == "text":
                        native_parts.append(
                            types.Part(
                                text=content.text
                            )
                        )
                    elif content.type == "media":  # Если пользователь приложил медиафайл к своему сообщению
                        media_parts = []
                        for asset in content.assets:
                            if asset.size_bytes < 20 * 1024 * 1024:  # Файлы меньше 20 Мб посылаем как строки
                                if asset.data_base64:
                                    raw_bytes = string_to_bytes(asset.data_base64)
                                else:
                                    raw_bytes = file_to_bytes(asset.local_path)
                                media_part = types.Part(
                                    inline_data=types.Blob(
                                        data=raw_bytes,
                                        mime_type=asset.mime_type
                                    )
                                )
                            else:  # Файлы больше 20 МБ добавляем как URI, ведущий на google cloud
                                media_asset = self._process_asset(asset)
                                media_part = types.Part(
                                    file_data=types.FileData(
                                        file_uri=media_asset.cloud_refs.genai.uri,
                                        mime_type=asset.mime_type
                                    )
                                )
                            media_parts.append(media_part)

        return native_history

    def _do_request(self, native_history, tools_definition):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=native_history,
            config=types.GenerateContentConfig(
                tools=tools_definition,
                system_instruction=self.system_prompt,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                thinking_config=self.thinking_config,
            ),
        )
        return response

    def generate(
            self,
            history: t.ChatData,
            tools_definition,
            tools_executable: Dict[str, Callable],
    ) -> tuple[t.ChatData, list[t.Message]]:
        native_history = self._convert_history_from_umf(history)

        response = self._do_request(native_history, tools_definition)

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
            elif part.inline_data:  # Текущие модели Gemini генерируют только изображение. Gemini Omni ещё недоступна в API, возможно генерация будет не в model.generate_content, как с nano banana, а в generate_video, как в Veo
                image = part.as_image()  # Возвращает объект types.Image. Может содержать либо gcs_uri, либо image_bytes
                image_id = message_helper.generate_id(settings.ASSET_ID_LEN)
                image_path = f"{settings.MEDIA_FOLDER}/{image_id}.png"  # В доках указано .png
                media_asset = t.Asset(
                    id=image_id,
                    type="image",
                    local_path=image_path,
                    mime_type="image/png",
                    size_bytes=0,
                )
                if image.image_bytes:
                    media_asset.data_base64 = bytes_to_string(image.image_bytes)
                elif image.gcs_uri:
                    media_asset.cloud_refs.genai.uri = image.gcs_uri
                content.append(
                    t.MediaContent(
                        type="media",
                        assets=[media_asset]
                    )
                )

        new_delta.append(
            t.Message(
                id=message_helper.generate_id(settings.MESSAGE_ID_LEN),
                role="assistant",
                content=content,
                timestamp=generate_timestamp(),
            )
        )

        # Затем цикл вызова инструментов с добавлением в историю.
        # Добавляется один message с ролью tool, и в контенте содержатся все результаты текущего раунда вызовов
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

        # Медиафайлы и метаданные не обрабатываются
        history.messages.extend(new_delta)
        return history, new_delta
