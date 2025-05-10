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

        Если `limit_bad` и `limit_good` совпадают и один is_*_soft=True, то будет два класса.

        ```
        'binary' -> 0: не-негатив, 1: негатив
        'ternary' -> 0: нейтрал, 1: позитив, 2: негатив
        ```

        """
        if limit_bad == limit_good and is_bad_soft and is_good_soft:
            raise ValueError(
                "Край пределов общий для двух классов в случае с общим значением края! is_bad_soft == is_good_soft == True!"
            )

        self.limit_bad = limit_bad
        self.limit_good = limit_good

        self.is_bad_soft = is_bad_soft
        self.is_good_soft = is_good_soft

        self.label_scheme = not (
            limit_bad == limit_good and any([is_bad_soft, is_good_soft])
        )
        """
        True - бинарная, False - тернарная
        """
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

        rdf = df.copy()

        fin_ratings = []
        labels = []

        for id in SERVICE_DICT.values():
            ratings = rdf[rdf["service_id"] == id]["rating"].tolist()

            scale = self.service_params.get(id, None)

            if scale == self.default_range or scale == None:
                fin_ratings.extend(ratings)
                continue

            for index, rating in enumerate(ratings):
                ratings[index] = self.__scale(rating, scale, self.default_range)

            fin_ratings.extend(ratings)

        for rating in fin_ratings:
            labels.append(self.__labeler(rating))

        rdf["label"] = labels

        return rdf

    def __labeler(self, rating) -> int:
        if (rating < self.limit_bad) or (self.is_bad_soft and rating <= self.limit_bad):
            # негатив
            return 1 + 1 * self.label_scheme

        if not self.label_scheme:
            # нейтраль (бинарная)
            return 0

        if (rating < self.limit_good) or (
            not self.is_good_soft and rating <= self.limit_good
        ):
            # нейтраль (тернарная)
            return 0

        return 1

    def __scale(
        self,
        rating: float,
        init_range: tuple[float, float],
        fin_range: tuple[float, float],
    ) -> float:
        if init_range == fin_range:
            return rating

        a, b = fin_range
        am, bm = init_range

        return a + (rating - am) / (bm - am) * (b - a)
