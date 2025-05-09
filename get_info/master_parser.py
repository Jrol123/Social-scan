"""
Пайплайн парсинга.
"""

from .abstract import Parser


class MasterParser:
    def __init__(
        self, *services: Parser, **parameters: dict[str, str | list[str] | int]
    ) -> None:
        """
        Сервисы передаются экземплярами!

        Параметры передаются лишь уникальные, по типу запросов, id и т. д., что будет отличаться от сервиса к сервису.

        Параметры должны быть вида `__class__.name__ = {"param": val}`
        """
        self.__serviceList = services
        self.__parseParameters = parameters

    def parse(self, **parameters) -> list[dict[str, str | int | float | None]]:
        """
        Параметры такие же, как и в parse у Parser
        """
        final_result = []
        for service in self.__serviceList:
            parseParameters = self.__parseParameters[service.__class__.__name__]
            result = service.parse(
                **parseParameters, **parameters
            )
            final_result.extend(result)
        return final_result
