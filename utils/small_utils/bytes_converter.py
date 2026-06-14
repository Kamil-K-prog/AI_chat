import base64

def string_to_bytes(input_string):
    """Преобразует Base64-строку в набор байт"""
    return base64.b64decode(input_string)

def bytes_to_string(input_bytes):
    """Преобразует набор байт в Base64-строку"""
    return base64.b64encode(bytes(input_bytes)).decode("utf-8")

def file_to_base64(file_path: str) -> str:
    """Преобразовывает файл в Base64-строку для отправки в gemini API напрямую, без загрузки в облако"""
    with open(file_path, 'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string.decode("utf-8")

def file_to_bytes(file_path: str) -> bytes:
    with open(file_path, 'rb') as file:
        return file.read()