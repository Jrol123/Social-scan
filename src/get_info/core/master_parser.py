"""
Пайплайн парсинга.
"""

from ..parsers.abstract import Parser, AsyncParser
from .config import GlobalConfig
from asyncio import run
from dataclasses import asdict


class MasterParser:
    def __init__(self, *parsers: Parser) -> None:
        """
        Сервисы передаются экземплярами!
        """
        self.parsers = parsers

    async def async_parse(
        self, global_params: GlobalConfig
    ) -> list[dict[str, str | int | float | None]]:
        """
        Параметры такие же, как и в parse у Parser.
        """
        results = []

        for parser in self.parsers:
            if isinstance(parser, AsyncParser):
                async with parser.client:
                    results += await parser.parse(global_params)
                # results += await parser.parse(global_params)
                continue
                
            results += parser.parse(global_params)

        return results

    def sync_parse(
        self, global_params: GlobalConfig
    ) -> list[dict[str, str | int | float | None]]:
        """
        Синхронная обёртка для совместимости.
        """
        return run(self.async_parse(global_params))
