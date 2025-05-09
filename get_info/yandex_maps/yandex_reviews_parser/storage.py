from dataclasses import dataclass
from typing import Union


@dataclass
class Review:
    name: str
    # icon_href: Union[str, None]
    date: float
    text: str
    stars: float
    answer: str


@dataclass
class Info:
    name: str
    rating: int
    stars: float
