import base64


def string_to_bytes(input_string):
    """Преобразует Base64-строку в набор байт"""
    return base64.b64decode(input_string)


def bytes_to_string(input_bytes):
    """Преобразует набор байт в Base64-строку"""
    return base64.b64encode(bytes(input_bytes)).decode("utf-8")
