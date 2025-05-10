"""
Пайплайн парсинга.
"""
import asyncio

from .abstract import Parser


class MasterParser:
    def __init__(
        self, *services: Parser, **parameters: dict[str, str | list[str] | int]
    ) -> None:
        """
        Сервисы передаются экземплярами!

        Параметры передаются лишь уникальные, по типу запросов, id и т. д., что будет отличаться от сервиса к сервису.

        Параметры должны быть вида `__class__.__name__ = {"param": val}`
        """
        self.__serviceList = services
        for service in self.__serviceList:
            service_name = service.__class__.__name__
            if service_name not in parameters:
                print(f"Нет параметров для сервиса '{service_name}'.")
        self.__parseParameters = parameters

    def parse(self, **parameters) -> list[dict[str, str | int | float | None]]:
        """
        Параметры такие же, как и в parse у Parser
        """
        final_result = []

        for service in self.__serviceList:
            service_name = service.__class__.__name__
            parseParameters = self.__parseParameters.get(service_name, {})

            try:
                result = service.parse(**parseParameters, **parameters)
                final_result.extend(result)
            except TypeError:
                raise Warning(service_name + " may be asynchronous, so it skips.")

        return final_result
