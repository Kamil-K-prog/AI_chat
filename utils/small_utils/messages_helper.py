import random
import string
from datetime import datetime, timezone


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


def generate_timestamp():
    now_utc = datetime.now(timezone.utc)
    iso_string = now_utc.isoformat()
    return iso_string


class MessageHelper:
    def __init__(self):
        self.ids = set()

    def generate_id(self, length):
        new_id = generate_random_string(length)
        while new_id in self.ids:
            new_id = generate_random_string(length)
        self.ids.add(new_id)
        return new_id


message_helper = MessageHelper()
