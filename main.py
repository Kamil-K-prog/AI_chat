from utils.tools_parser import ToolsParser # Парсер для инструментов
import utils.tools # Импорт, чтобы сработали декораторы # noqa: F401

# --- Инструменты для нейросети ---
tools_parser = ToolsParser()

print(tools_parser.get_tools_json_openai())