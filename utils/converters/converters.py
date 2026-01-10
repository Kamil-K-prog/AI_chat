from typing import List, Dict, Any
import utils.types as t
from google.genai import types
from openai.types.chat import ChatCompletion

"""
Функции для конвертации унифицированного формата в genai или openai совместимый (для всей истории), 
и наоборот (для новых сообщений).
"""


def from_history_to_openai(history: t.ChatHistory) -> List[Dict[str, Any]]:
    pass


def from_history_to_genai(history: t.ChatHistory) -> List[types.Content]:
    # TODO: сделать так, чтобы небольшие файлы из Base64(поддерживаемый OpenAI) перекодировались в Bytes(поддерживаемый GenAI). С большими файлами логика другая - при препроцессинге истории сообщений в самой модели оставшиеся большие файлы загружаются в облако, и path заменяется на URI
    res = []
    for message in history.messages:
        content = message.content
        parts = []
        role = ""

        if message.role == "user":
            role = "user"
            for part in content:
                if isinstance(part, t.TextData):
                    parts.append(types.Part.from_text(text=part.data))
                elif isinstance(part, t.MediaData):
                    for media_item in part.data:
                        if media_item.type == "image":
                            mime_type = "image/png"
                        elif media_item.type == "audio":
                            mime_type = "image/png"
                        else:  # video
                            mime_type = "image/png"
                        if media_item.source_type == "base64":
                            parts.append(types.Part.from_bytes(data=media_item.data, mime_type=mime_type))
                        elif media_item.source_type == "path" or media_item.source_type == "url":
                            parts.append(types.Part.from_uri(file_uri=media_item.data, mime_type=mime_type))

        elif message.role == "model":
            role = "model"
            for part in content:
                # Игнорируем thoughts. Из UI они могут не приходить, а thoughts, которые модель будет генерировать во время цепочки вызова функций, будут внутри модели, и вернутся вместе с дельтой из функции generate(). Т.е. UI их получит и сохранит, а модели старые CoT поступать не будут

                if isinstance(part, t.TextData):
                    parts.append(types.Part.from_text(text=part.data))
                elif isinstance(part, t.MediaData):
                    for media_item in part.data:
                        if media_item.type == "image":
                            mime_type = "image/png"
                        elif media_item.type == "audio":
                            mime_type = "audio/mp3"
                        else:  # video
                            mime_type = "video/mp4"
                        if media_item.source_type == "base64":
                            parts.append(types.Part.from_bytes(data=media_item.data, mime_type=mime_type))
                        elif media_item.source_type == "path" or media_item.source_type == "url":
                            parts.append(types.Part.from_uri(file_uri=media_item.data, mime_type=mime_type))
                elif isinstance(part, t.ToolCallsData):
                    for tool_call in part.data:
                        parts.append(types.Part.from_function_call(name=tool_call.tool_name, args=tool_call.args))

        elif message.role == "tool_responses":
            role = "tool"
            for part in content:
                if isinstance(part, t.ToolResponseItem):
                    if not part.error:
                        parts.append(
                            types.Part.from_function_response(name=part.tool_name, response={'result': part.response}))
                    else:
                        parts.append(
                            types.Part.from_function_response(name=part.tool_name, response={'error': part.response}))

        # Добавляем новый элемент в историю
        res.append(types.Content(
            role=role,
            parts=parts
        ))
    return res


def from_openai_to_message(response: ChatCompletion) -> List[t.Message]:
    pass


def from_genai_to_message(response: types.GenerateContentResponse) -> List[t.Message]:
    pass
