from dataclasses import dataclass
from datetime import datetime

from ..parsers.abstract import ParserConfig


@dataclass
class MasterParserConfig(ParserConfig):
    min_date: datetime | int = datetime(1970, 1, 16)
    max_date: datetime | int = datetime.now()
    sort_type: str = "ascending"
    count_items: int = -1
