import base64
import os


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
