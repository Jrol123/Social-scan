from ...abstract import ParserConfig
from dataclasses import dataclass
import logging


@dataclass
class GoogleMapsConfig(ParserConfig):
    q: str | list[str]

    def __init__(self, q: str | list[str]):
        if not (
            isinstance(q, str)
            or (isinstance(q, list) and isinstance(q[0], str) and len(q) == 2)
        ):
            raise ValueError(f"Был введён неправильный вид запроса для q")
        self.q = q
        self.collect_extra = False
        self.wait_load = 60

        # except ValueError:
        #     logging.critical(f"Был введён неправильный вид запроса")
