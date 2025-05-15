from dataclasses import dataclass
from datetime import datetime

from ..abstract import GlobalConfig


@dataclass
class MasterConfig(GlobalConfig):
    """
    Класс, содержащий глобальные параметры для MasterParser.

    Args:
        min_date (datetime | int): Время самого раннего сообщения в формате datetime или timestamp. Defaults to ```datetime(1970, 1, 16)```.
        max_date (datetime | int): Время самого позднего сообщения в формате datetime или timestamp. Defaults to ```datetime.now()``` (now считается от момента генерации конфига). #! TODO: сделать так, чтобы пересчитывалось каждый раз при запуске парсинга
        sort_type (str): Вид сортировки. Зависят от сервиса (см. документацию к каждому сервису отдельно). Defaults to `ascending`.
        count_items (int): Максимальное количество возвращаемых элементов. Для получения всех используется значение -1. Defaults to -1

    """

    min_date: datetime | int
    max_date: datetime | int
    sort_type: str
    count_items: int

    def __init__(
        self,
        min_date: datetime | int = datetime(1970, 1, 16),
        max_date: datetime | int = None,
        sort_type: str = "ascending",
        count_items: int = -1,
    ) -> None:
        self.min_date = min_date
        self.max_date = self._check_data(max_date)
        self.sort_type = sort_type
        self.count_items = count_items
