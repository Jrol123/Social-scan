from ...abstract import ParserConfig
from dataclasses import dataclass


@dataclass
class OtzovikConfig(ParserConfig):
    q: str

    def __init__(self, q: str):
        self.q = q
