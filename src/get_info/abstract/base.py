from abc import ABC, abstractmethod
from datetime import datetime
from typing import Type

from ...abstract import Config


# TODO: По-хорошему нужно было сделать отдельно конфиги для карт, отдельно для соц-сетей, чтобы разделить неиспользуемые атрибуты (по-типу сортировки), но...


class GlobalConfig(Config):
    """
    Класс, содержащий общие параметры для всех Parser.
    """


class ParserConfig(Config):
    """
    Класс, содержащий локальные параметры для Parser.

    На вход подаются параметры, которые индивидуальны для каждого парсера.

    Args:
        q (str | list[str]): Информация, необходимая для поиска объекта в сервисе.

    """

    def __init__(self, q: str | list[str]) -> None:
        self.q = q


class Parser(ABC):
    def __init__(self, service_id: int, config: ParserConfig = None):
        """
        Парсер.

        Args:
            service_id (int): Индекс сервиса.

        """
        self.service_id = service_id
        self.config = config

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
            raise ValueError("Доступна конвертация только в int и `datetime`")
        if isinstance(date_datetime, final_type):
            return date_datetime
        if isinstance(date_datetime, datetime):
            return int(date_datetime.timestamp())
        return datetime.fromtimestamp(date_datetime)

    @abstractmethod
    def parse(
        self, global_config: GlobalConfig
    ) -> list[dict[str, str | int | float | None]]:
        """
        Получение информации с сервиса по запросу.

        На вход получаются параметры, которые будут общими для всех парсеров.

        Возможно получение пустого конфига и автогенерация полного.

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

    def update_config(self, config: ParserConfig) -> None:
        self.local_config = config


class AsyncParser(Parser):
    """
    Абстрактный парсер.
    """

    client: object

    @abstractmethod
    async def parse(
        self, global_config: GlobalConfig
    ) -> list[dict[str, str | int | float | None]]:
        pass
