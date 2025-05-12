from abc import ABC, abstractmethod
from datetime import datetime
from typing import Type


class Parser(ABC):
    def __init__(self, service_id: int):
        self.service_id = service_id

    def _date_convert(
        self, date_datetime: datetime | int, final_type: Type[int] | Type[datetime]
    ) -> int | datetime:
        """
        Конвертация даты.

        Args:
            date_datetime (datetime | int): Время.
            final_type (datetime | int): Формат, который нужно получить.

        Returns:
            int|datetime: Время в указанном формате.
        """
        if final_type not in (int, datetime):
            raise ValueError("Достпна конвертация только в int и `datetime`")
        if isinstance(date_datetime, final_type):
            return date_datetime
        if isinstance(date_datetime, datetime):
            return int(date_datetime.timestamp())
        return datetime.fromtimestamp(date_datetime)

    @abstractmethod
    def parse(
        self,
        q: str | list[str],
        min_date: datetime | int = datetime(1970, 1, 16),
        max_date: datetime | int = datetime.now(),
        sort_type: str = "ascending",
        count_items: int = -1,
    ) -> list[dict[str, str | int | float | None]]:
        """
        Получение информации с сервиса по запросу.

        Args:
            q (str | list[str]): Информация, необходимая для поиска объекта в сервисе.
            min_date (datetime | int): Время самого раннего сообщения в формате datetime или timestamp. Defaults to ```datetime.min```.
            max_date (datetime | int): Время самого позднего сообщения в формате datetime или timestamp. Defaults to ```datetime.now()```.
            sort_type (str): Вид сортировки. Зависят от сервиса (см. документацию к каждому сервису отдельно). Defaults to `ascending`.
            count_items (int): Максимальное количество возвращаемых элементов. Для получения всех используется значение -1. Defaults to -1

        Returns:
            list[dict[str,str|int|float|None]]: Список сообщений. Каждое сообщение - словарь с данными. Структура сообщения:
            ```
            {
                "service_id" (int): Внутренний индекс сервиса
                "name" (str): Имя пользователя. Для Telegram и VK хранить id пользователя.
                "additional_id" (str | None): Дополнительный идентификатор для уточнения сообщения (пример: канал в Telegram).
                "date" (int): Дата в формате timestamp.
                "rating" (float | None): Рейтинг (1.0-5.0, если есть, иначе None).
                "text" (str): Текст отзыва.
                "answer" (str | None): Ответ на отзыв (если присутствует).
            }
            ```

        Examples:
            **Вывод с Google карт:**
            ```
            [
                {
                    "service_id": 0,
                    "name": "Иван Иванов",
                    "additional_id": None,
                    "date": 1678901234,
                    "rating": 5.0,
                    "text": "Отличный сервис!",
                    "answer": "Спасибо за отзыв!",
                },
                ...
            ]
            ```

            **Вывод с Telegram:**
            ```
            [
                {
                    "service_id": 4,
                    "name": "nickname",
                    "additional_id": "457",
                    "date": 1678901234,
                    "rating": None,
                    "text": "Отличный сервис!",
                    "answer": None,
                },
                ...
            ]
            ```

        """
        pass


class AsyncParser(Parser):
    @abstractmethod
    async def parse(
        self,
        q: str | list[str],
        min_date: datetime | int = datetime(1970, 1, 16),
        max_date: datetime | int = datetime.now(),
        sort_type: str = "ascending",
        count_items: int = -1,
    ) -> list[dict[str, str | int | float | None]]:
        pass


class ParserConfig(ABC):
    pass
