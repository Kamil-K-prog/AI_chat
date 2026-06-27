import os
import time
import mimetypes
from typing import Dict, Callable, List
import filetype

from google import genai
from google.genai import types, errors

from config import settings
from utils.small_utils import (
    message_helper,
    generate_timestamp,
    string_to_bytes,
    bytes_to_string,
    file_to_bytes
)
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

    def _save_media_from_gcs(self, uri, local_path) -> None:
        """
        ЗАГОТОВКА
        Загружает медиафайл из GCS и сохраняет его
        :param uri: ссылка для получения файла
        :param local_path: путь для сохранения
        :return: None
        """
        pass

    def _process_asset(self, asset: t.Asset) -> t.Asset:
        """
        Загружает ассет в Google Files API (или берёт уже загруженный) и возвращает
        Asset с заполненными cloud_refs.genai. Для видео/аудио дожидается ACTIVE.

        TODO: если ассет был загружен в GCS, то обновлять его в УФС, чтобы избежать повторной загрузки
        """
        if asset.cloud_refs is None:
            asset.cloud_refs = t.CloudRefs(genai=t.CloudRef())
        elif asset.cloud_refs.genai is None:
            asset.cloud_refs.genai = t.CloudRef()

        filename = asset.cloud_refs.genai.filename
        try:
            if filename:
                file = self.client.files.get(name=filename)
                asset.cloud_refs.genai.file_object = file
                return asset
        except errors.APIError as e:
            if e.code != 404:
                raise

        file = self.client.files.upload(file=asset.local_path)

        while file.state.name == "PROCESSING":
            time.sleep(5)
            file = self.client.files.get(name=file.name)
        if file.state.name == "FAILED":
            raise RuntimeError(f"Обработка файла в Google Files API завершилась ошибкой: {file.error}")

        asset.cloud_refs.genai.filename = file.name
        asset.cloud_refs.genai.file_object = file
        asset.cloud_refs.genai.uri = file.uri
        asset.cloud_refs.genai.expires_at = file.expiration_time
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
            elif message.role == "assistant":
                preserved_thought_signature = None
                for content in message.content:
                    if self.thinking_config and content.type == "thought":
                        if content.signature:  # Если ответ от модели genai, то есть подпись, и эту CoT можно подать на вход.
                            # Если мысли не подписаны, то API вернет ошибку
                            # --- ПРОБЛЕМА --- Начиная с Gemini 3 если не вернуть мысли в цикле ReAct, то API вернёт ошибку 400
                            # https://ai.google.dev/gemini-api/docs/thought-signatures?hl=ru#model-behavior
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
                                "output": content.tool_result.text_content,
                                "error": str(content.tool_result.is_error),
                            },
                        )
                    )
                    for asset in content.assets or []:
                        if asset.size_bytes < 20 * 1024 * 1024:  # Файлы меньше 20 Мб посылаем как строки
                            if asset.data_base64:
                                raw_bytes = string_to_bytes(asset.data_base64)
                            else:
                                raw_bytes = file_to_bytes(asset.local_path)
                            function_response_part = types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    data=raw_bytes,
                                    mime_type=asset.mime_type,
                                )
                            )
                        else:  # Файлы больше 20 МБ добавляем как URI, ведущий на google cloud
                            media_asset = self._process_asset(asset)
                            function_response_part = types.FunctionResponsePart(
                                file_data=types.FunctionResponseFileData(
                                    file_uri=media_asset.cloud_refs.genai.uri,
                                    mime_type=asset.mime_type,
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
                        native_parts.extend(media_parts)
                native_history.append(
                    types.Content(
                        role="user",
                        parts=native_parts,
                    )
                )

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
            extra_body: dict,
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
            elif part.inline_data:  # Текущие модели Gemini генерируют только изображение.
                image = part.as_image()  # Может содержать либо gcs_uri, либо image_bytes
                if image:
                    image_id = message_helper.generate_id(settings.ASSET_ID_LEN)
                    os.makedirs(settings.MEDIA_FOLDER, exist_ok=True)
                    image_path = f"{settings.MEDIA_FOLDER}/{image_id}.png"  # В доках указано .png
                    media_asset = t.Asset(
                        id=image_id,
                        type="image",
                        local_path=image_path,
                        mime_type=image.mime_type or "image/png",
                    )
                    if image.image_bytes:
                        image.save(image_path)
                        media_asset.size_bytes = len(image.image_bytes)
                    # Код ниже закомментирован, так как в документации нет слов о том,
                    # что сгенерированное nano banana изображение может НЕ содержаться в виде байтов и его надо скачивать
                    # elif image.gcs_uri:
                    #     media_asset.cloud_refs = t.CloudRefs(
                    #         genai=t.CloudRef(uri=image.gcs_uri)
                    #     )
                    #     media_asset.size_bytes = count_file_size(image_path)
                    content.append(
                        t.MediaContent(
                            type="media",
                            assets=[media_asset]
                        )
                    )
                else:  # Непонятно, что ещё кроме изображения может вернуть модель
                    pass

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
            tool_result_asset = None
            try:
                current_tool = tools_executable[tool_call.name]
                # Если функция возвращает медиа
                if getattr(current_tool, "returns_media", False):
                    tool_result = tools_executable[tool_call.name](**tool_call.args)

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
                    elif mime_type.startswith("text/") or mime_type.startswith("application/"):
                        asset_type = "document"
                    else:
                        raise TypeError(f"Функция {tool_call.name} сгенерировала файл неподдерживаемого типа")

                    # Сохраняем медиа файл
                    asset_extension = mimetypes.guess_extension(mime_type) or ".bin"
                    asset_local_path = f"{settings.MEDIA_FOLDER}/{asset_id}{mimetypes.guess_extension(mime_type)}"
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
                    tool_result_str = tools_executable[tool_call.name](**tool_call.args)

            except Exception as e:
                is_error = True
                tool_result_str = str(e)

            results.append(
                t.ToolResultContent(
                    type="tool_result",
                    tool_result=t.ToolResult(
                        id=tool_call.id or tool_call_id_fallback,
                        name=tool_call.name,
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

        # Медиафайлы и метаданные не обрабатываются
        history.messages.extend(new_delta)
        return history, new_delta
