from abc import ABC, abstractmethod


class Parser(ABC):
    @abstractmethod
    def parse(self) -> dict[str, str | int | None]:
        """
        Получение информации с сервиса по запросу

        Returns:
            dict[str,str|int|None]:
                {
                    "name": user (str),
                    "date": date (timestamp),
                    "rating": rating (5 | None),
                    "text": text (str)
                    "answer": response (str | None)
                }
        """
