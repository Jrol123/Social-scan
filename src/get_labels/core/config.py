from pandas import DataFrame
from ...abstract import GlobalConfig


class MasterTransformerConfig(GlobalConfig):
    RATING_SERVICE = [
        0,
        1,
        2,
    ]
    """Сервисы с рейтингом"""
    #! TODO: считывать из .txt файла

    def __init__(self, df: DataFrame) -> None:
        super().__init__()
        self.rDf = df[df["service_id"].isin(self.RATING_SERVICE)]
        self.sDf = df[~df["service_id"].isin(self.RATING_SERVICE)]

    # Возможно, надо будет добавить доп. инфу в конфиг (.txt).
