from dataclasses import dataclass
from datetime import datetime

from ..abstract import GlobalConfig


@dataclass
class MasterParserConfig(GlobalConfig):

    SORT_TYPES = ["rating_ascending", "rating_descending", "date_descending", "default"]

    GET_ALL_ITEMS = -1

    def __init__(
        self,
        min_date: datetime | int = datetime(1970, 1, 16),
        max_date: datetime | int | None = None,
        sort_type: str = "ascending",
        count_items: int = GET_ALL_ITEMS,
    ) -> None:
        """
        Класс, содержащий глобальные параметры для MasterParser.

        Args:
            min_date (datetime | int): Время самого раннего сообщения в формате datetime или timestamp. Defaults to ```datetime(1970, 1, 16)```.
            max_date (datetime | int): Время самого позднего сообщения в формате datetime или timestamp. Defaults to ```datetime.now()``` (now считается от момента генерации конфига). #! TODO: сделать так, чтобы пересчитывалось каждый раз при запуске парсинга
            sort_type (str): Вид сортировки. Зависит от сервиса (см. документацию к каждому сервису отдельно). Defaults to `ascending`.
            count_items (int): Максимальное количество возвращаемых элементов. Для получения всех используется значение {self.GET_ALL_ITEMS}. Defaults to {self.GET_ALL_ITEMS}

        """
        self.min_date = min_date
        """Время самого раннего сообщения в формате datetime или timestamp."""
        self.max_date = self._check_data(max_date)
        """Время самого позднего сообщения в формате datetime или timestamp."""
        assert sort_type in self.SORT_TYPES
        self.sort_type = sort_type
        """Вид сортировки."""
        self.count_items = count_items
        """Максимальное количество возвращаемых элементов. Для получения всех используется значение {self.GET_ALL_ITEMS}."""
