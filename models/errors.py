"""Исключения предметной области (domain-specific) для слоя моделей."""


class ModelLayerError(Exception):
    """Базовое исключение для ошибок, возникающих в абстрактном слое моделей."""


class UnknownModelError(ModelLayerError, KeyError):
    """Вызывается, когда имя запрашиваемой модели отсутствует в каталоге."""


class UnsupportedProviderError(ModelLayerError, ValueError):
    """Вызывается, когда спецификация каталога ссылается на провайдера без зарегистрированной фабрики."""


class UnsupportedAssetError(ModelLayerError, ValueError):
    """Вызывается, когда ассет не может быть подготовлен для выбранной модели/провайдера."""


class NativeResponseParseError(ModelLayerError, ValueError):
    """Вызывается, когда ответ провайдера не может быть преобразован обратно в УФС."""
