"""
Утилиты для нормализации медиафайлов, возвращаемых инструментами.

ToolRunner не должен содержать низкоуровневую логику определения MIME-типов и сохранения файлов;
эта зона ответственности изолирована здесь, чтобы ранер оставался координирующим (оркестрационным) классом.
"""
import mimetypes

import filetype
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Literal, Tuple, Dict
from pathlib import Path

import utils.types as t
from config import settings
from models.converters import media
from utils.small_utils import message_helper, bytes_to_string, generate_absolute_path


@dataclass(frozen=True)
class NormalizedMediaResult:
    """Провайдеро-независимое представление результата инструмента, генерирующего медиа."""
    data: bytes
    asset_type: Literal["image", "audio", "video", "document"]
    mime_type: str


class ToolMediaResultBuilder:
    """Нормализует и сохраняет медиафайлы, возвращаемые инструментами, в объекты Asset УФС."""

    def normalize(self, media_bytes_arr: List[bytes], mime_type_str: Optional[str] = None) -> List[NormalizedMediaResult]:
        """
        Преобразовать поддерживаемые форматы возвращаемых значений инструментов в один явный объект данных.

        Поддерживаемые форматы определяются соглашением проекта для инструментов, помеченных
        как ``returns_media``.

        :param media_bytes_arr: Сырое значение, возвращённое функцией инструмента
        :param mime_type_str: Если у функции определен MIME тип возвращаемого значения
        :return: Нормализованные медиа-данные и текстовое описание.
        :raises TypeError: Если формат возвращаемого значения не поддерживается.
        """
        res = []

        for media_bytes in media_bytes_arr:
            # Угадываем MIME тип
            kind = filetype.guess(media_bytes)
            if not kind and not mime_type_str:
                raise TypeError(f"Функция сгенерировала файл неизвестного типа")
            mime_type = mime_type_str or kind.mime

            # Определяем тип для ассета
            if mime_type.startswith("image/"):
                asset_type = "image"
            elif mime_type.startswith("audio/"):
                asset_type = "audio"
            elif mime_type.startswith("video/"):
                asset_type = "video"
            elif mime_type.startswith("text/") or mime_type.startswith("application/"):
                asset_type = "document"
            else:
                raise TypeError(f"Функция сгенерировала файл неподдерживаемого типа")

            res.append(NormalizedMediaResult(
                data=media_bytes,
                asset_type=asset_type,
                mime_type=mime_type,
            ))

        return res

    def build_assets(self, media_bytes_arr: List[bytes], mime_type_str: Optional[str] = None) -> List[t.Asset]:
        """
        Сохранить результат инструмента, генерирующего медиа, и вернуть соответствующий Asset УФС.

        :param tool_result_dict: все данные о результате выполнения инструмента
        :return: Массив ассетов УФС
        """
        res = []
        normalized_assets = self.normalize(media_bytes_arr=media_bytes_arr, mime_type_str=mime_type_str)

        for asset in normalized_assets:
            asset_id = message_helper.generate_id(settings.ASSET_ID_LEN)
            file_extension = mimetypes.guess_extension(asset.mime_type) or ".bin"
            absolute_asset_path = generate_absolute_path(asset_id, file_extension)
            with open(absolute_asset_path, mode="wb") as f:
                f.write(asset.data)
            size_bytes = len(asset.data)
            res.append(t.Asset(
                id=asset_id,
                type=asset.asset_type,
                absolute_path=absolute_asset_path,
                mime_type=asset.mime_type,
                size_bytes=size_bytes,
                data_base64=bytes_to_string(asset.data) if size_bytes < 20 * 1024 * 1024 else None
            ))

        return res

    def process_function_response(self, tool: Callable, tool_result: Any) -> Tuple[str, List[t.Asset]]:
        tool_result_dict = {
            "text": "<mock_text>Function run complete, but no text generated</mock_text>",
            "media_bytes_arr": [],
            "mime_type_str": getattr(tool, "mime_type", None),
        }
        match tool_result:
            case bytes(media_bytes):
                tool_result_dict["media_bytes_arr"] = [media_bytes]
            case list(media_bytes_arr):
                tool_result_dict["media_bytes_arr"] = media_bytes_arr
            case (str() | None as text, bytes() | list() as media, str() | None as mime):
                tool_result_dict["text"] = text
                if isinstance(media, list):
                    tool_result_dict["media_bytes_arr"].extend(media)
                else:
                    tool_result_dict["media_bytes_arr"].append(media)
                tool_result_dict["mime_type_str"] = mime or tool_result_dict["mime_type_str"]
            case (bytes() | list() as media, str() | None as mime):
                if isinstance(media, list):
                    tool_result_dict["media_bytes_arr"].extend(media)
                else:
                    tool_result_dict["media_bytes_arr"].append(media)
                tool_result_dict["mime_type_str"] = mime or tool_result_dict["mime_type_str"]
            case (str() | None as text, bytes() | list() as media):
                tool_result_dict["text"] = text
                if isinstance(media, list):
                    tool_result_dict["media_bytes_arr"].extend(media)
                else:
                    tool_result_dict["media_bytes_arr"].append(media)
            case _:
                raise TypeError("Функция сгенерировала результат некорректного типа")



        assets = self.build_assets(media_bytes_arr=tool_result_dict["media_bytes_arr"], mime_type_str=tool_result_dict["mime_type_str"])
        return (tool_result_dict["text"], assets)