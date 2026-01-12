from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]):
        pass