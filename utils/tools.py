# Файл, содержащий функции, доступные для выполнения нейросетью.
# Описание формируется парсером из docstring

from utils.tools_parser import register_tool

# @register_tool Функция-пример не будет зарегистрирована
def foo_func(first: int, second: float=2) -> float:
    """
    Короткое описание в одну строку.

    Длинное описание. Может содержать в себе несколько строк.
    Оба этих описания будут поданы модели.

    :param first: Первый параметр.
    :param second: Второй параметр.
    :return: Сумма первого и второго параметра
    """
    return first + second

@register_tool
def bar_func(first: float, second: float) -> float:
    """
    Функция складывает два числа

    Складывает целочисленное число и число с плавающей запятой, и возвращает их сумму
    :param first: Первый параметр
    :param second: Второй параметр
    :return: Сумма первого и второго параметра
    """
    return first + second