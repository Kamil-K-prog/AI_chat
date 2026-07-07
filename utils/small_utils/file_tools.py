import base64
import os
from config import settings
from pathlib import Path

def file_to_base64(file_path: str) -> str:
    """Преобразовывает файл в Base64-строку"""
    with open(file_path, 'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string.decode("utf-8")


def file_to_bytes(file_path: str) -> bytes:
    """Считывает файл бинарно"""
    with open(file_path, 'rb') as file:
        return file.read()


def count_file_size(file_path: str) -> int:
    """Возвращает размер файла в байтах"""
    return os.path.getsize(file_path)

def generate_absolute_path(asset_id: str, file_extension: str) -> str:
    absolute_asset_path = Path(settings.PROGRAM_DIR) / settings.MEDIA_FOLDER / asset_id / file_extension
    absolute_asset_path.parent.mkdir(parents=True, exist_ok=True)
    return absolute_asset_path