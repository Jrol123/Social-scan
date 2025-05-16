from ....abstract import ParserConfig
from dataclasses import dataclass
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="parsing.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)


@dataclass
class YandexMapsConfig(ParserConfig):
    q: int
    def __init__(self, q: str | int):
        try:
            self.q: int = int(q)
        except ValueError:
            logging.critical(f"Был введён неправильный вид запроса")
