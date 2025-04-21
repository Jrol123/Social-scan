from abc import ABC, abstractmethod
from datetime import datetime


class Parser(ABC):
    def __init__(self, service_id: int):
        self.service_id = service_id

    def __date_convert(self, date_datetime: datetime) -> int:
        """
        Конвертация даты в формат timestamp.
        
        Требуется для некоторых парсеров

        Args:
            date_datetime (datetime): Время в формате datetime.

        Returns:
            int: Время в формате timestamp.
        """
        return int(date_datetime.timestamp())

    @abstractmethod
    def parse(
        self,
        q: str | list[str],
        oldest_date: datetime = datetime.min,
        count_items: int = -1,
    ) -> list[dict[str, str | int | float | None]]:
        """
        Получение информации с сервиса по запросу.

        Args:
            q (str | list[str]): Информация, необходимая для поиска объекта в сервисе.
            oldest_date (datetime): Время самого позднего сообщения в формате datetime. Defaults to ```datetime.min```.
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
                    "service_id": 0
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
                    "service_id": 4
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
