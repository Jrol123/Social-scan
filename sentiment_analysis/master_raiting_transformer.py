import pandas as pd

# TODO: В MasterParser нужно будет конвертировать числовые id сервисов

SERVICE_DICT = {
    "GoogleMapsParser": 0,
    "YandexMapsParser": 1,
    "OtzovikParser": 2,
}


class MasterRaitingTransformer:
    def __init__(
        self,
        limit_bad: float = 3.0,
        limit_good: float = 3.0,
        is_bad_soft: bool = True,
        is_good_soft: bool = False,
        default_range: tuple[float, float] = (1.0, 5.0),
        **count_range: tuple[float, float]
    ) -> None:
        """
        `count_range` подаётся в виде: НазваниеКлассаПарсера=tuple[float, float]

        `limit_bad` считается по-умолчанию нестрого, тогда как `limit_good` строго.

        """
        self.limit_bad = limit_bad
        self.limit_good = limit_good

        self.is_bad_soft = is_bad_soft
        self.is_good_soft = is_good_soft

        self.default_range = default_range

        self.service_params: dict[int, tuple[float, float]] = {}
        for name, rng in count_range.items():
            self.service_params[SERVICE_DICT[name]] = rng

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Трансформирование рейтинга в label

        Args:
            df (pd.DataFrame): Должен быть ТОЛЬКО С сообщениями с рейтингом.
        """
        # 1 - Негативные отзывы

        sdf = df[["service_id", "rating"]]
        
        for id in SERVICE_DICT.values():
            sdf["service_id" == id]

    def __scale(
        self,
        rating: float,
        init_range: tuple[float, float],
        fin_range: tuple[float, float],
    ):
        if init_range == fin_range:
            return rating

        a, b = fin_range
        am, bm = init_range

        return a + (rating - am) / (bm - am) * (b - a)
