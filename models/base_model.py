from abc import ABC, abstractmethod
import utils.types as t

class BaseModel(ABC):
    @abstractmethod
    def generate(self, history: t.ChatData, tools_definition, tools_executable, extra_body: dict) -> tuple[t.ChatData, list[t.Message]]:
        pass