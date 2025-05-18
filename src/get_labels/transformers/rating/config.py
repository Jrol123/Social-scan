from ....abstract import Config

#! TODO: В MasterParser нужно будет конвертировать числовые id сервисов
# Можно будет сделать получение name: id из того же txt, из какого будет получение id для парсеров.


class MasterRatingConfig(Config):
    SERVICE_DICT = {
        "GoogleMapsParser": 0,
        "YandexMapsParser": 1,
        "OtzovikParser": 2,
    }
    """Сервисы -> id"""

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
        #! Костыль со вводом размахов сервисов через названия их классов. Не придумал, как можно вводить сразу числовые значения (id сервисов)
        super().__init__()
        if limit_bad == limit_good and is_bad_soft and is_good_soft:
            raise ValueError(
                "Край пределов общий для двух классов в случае с общим значением края! is_bad_soft == is_good_soft == True!"
            )
        if limit_bad > limit_good:
            raise ValueError(
                "Неправильно заданы края рейтингов! `limit_bad` должна быть <= `limit_good`!"
            )

        self.limit_bad = limit_bad
        """Граница негатива"""
        self.limit_good = limit_good
        """Граница позитива"""

        self.is_bad_soft = is_bad_soft
        """Нестрогое ли условие на негатив"""
        self.is_good_soft = is_good_soft
        """Нестрогое ли условие на позитив"""

        self.label_scheme = not (
            limit_bad == limit_good and any([is_bad_soft, is_good_soft])
        )
        """
        Схема разметки.
        
        True - бинарная, False - тернарная.
        """
        self.default_range = default_range
        """Стандартный размах оценок."""

        self.service_params: dict[int, tuple[float, float]] = {}
        for name, rng in count_range.items():
            self.service_params[SERVICE_DICT[name]] = rng
            """Список размахов оценок для сервисов."""
