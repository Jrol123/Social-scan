"""
Пайплайн парсинга.
"""

from .abstract import Parser


class MasterParser:
    def __init__(
        self,
        *services: Parser,
        **parameters: dict[str, dict[str, str | list[str] | int]]
    ) -> None:
        """
        Сервисы передаются экземплярами!

        Параметры передаются лишь уникальные, по типу запросов, id и т. д., что будет отличаться от сервиса к сервису.
        """
        self.__serviceList = services
        self.__parseParameters = parameters

    def parse(self, **parameters) -> list[dict[str, str | int | float | None]]:
        """
        Параметры такие же, как и в parse у Parser
        """
        final_result = []
        for service in self.__serviceList:
            result = service.parse(**parameters)
            final_result.extend(result)
        return final_result
