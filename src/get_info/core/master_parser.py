"""
Пайплайн парсинга.
"""

from ..abstract import Parser, AsyncParser
from .config import MasterParserConfig
from asyncio import run


class MasterParser:
    def __init__(self, *parsers: Parser) -> None:
        """
        Сервисы передаются экземплярами!
        """
        self.parsers = parsers

    async def async_parse(
        self, global_params: MasterParserConfig
    ) -> list[dict[str, str | int | float | None]]:
        """
        Параметры такие же, как и в parse у Parser.
        """
        results = []

        #! TODO: Сделать сохранение промежуточных результатов в папке tmp. После полного парсинга собирать файлы и удалять эту папку вместе с содержимым.

        for parser in self.parsers:
            print(parser.__class__.__name__)
            if isinstance(parser, AsyncParser):
                # async with parser.client:
                results += await parser.parse(global_params)
                # results += await parser.parse(global_params)
                continue

            results += parser.parse(global_params)

        return results

    def sync_parse(
        self, global_params: MasterParserConfig
    ) -> list[dict[str, str | int | float | None]]:
        """
        Синхронная обёртка для совместимости.
        """
        return run(self.async_parse(global_params))
