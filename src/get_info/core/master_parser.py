"""
Пайплайн парсинга.
"""

from ..parsers.abstract import Parser, AsyncParser, ParserConfig
from asyncio import run
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ParserInstance:
    parser: Parser | AsyncParser
    config: ParserConfig


class MasterParser:
    def __init__(self, parsers: list[ParserInstance]) -> None:
        """
        Сервисы передаются экземплярами!

        Параметры передаются лишь уникальные, по типу запросов, id и т. д., что будет отличаться от сервиса к сервису.

        Параметры должны быть вида `__class__.__name__ = {"param": val}`
        """
        self.parsers = parsers

    async def async_parse(
        self, **global_params: str | list[str] | int | datetime
    ) -> list[dict[str, str | int | float | None]]:
        """
        Параметры такие же, как и в parse у Parser.
        """
        results = []

        for instance in self.parsers:
            instance.config.apply(instance.parser)

            if isinstance(instance.parser, AsyncParser):
                results += await instance.parser.parse(**global_params)
                continue
            results += instance.parser.parse(**global_params)

        return results

    def sync_parse(
        self, **global_params: str | list[str] | int | datetime
    ) -> list[dict[str, str | int | float | None]]:
        """
        Синхронная обёртка для совместимости.
        """
        return run(self.async_parse(**global_params))
