from typing import Callable
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, tools_definition, tools_executable: dict[str, Callable]):
        self.tools_definition = tools_definition
        self.tools_executable = tools_executable

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]):
        pass