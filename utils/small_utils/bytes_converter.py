import base64

def string_to_bytes(input_string):
    return base64.b64decode(input_string)

def bytes_to_string(input_bytes):
    return base64.b64encode(bytes(input_bytes)).decode("utf-8")