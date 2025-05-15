from ..abstract import ParserConfig
from dataclasses import dataclass


@dataclass
class TelegramConfig(ParserConfig):
    q: str
    channels_list: str | list[str]
    """Путь до файла со списком каналов / список каналов"""
    wait_sec: int

    def __init__(
        self, q: str, channel_list: str | list[str] = "channel_list.txt", wait_sec: int = 1
    ):
        self.q = q
        self.channels_list = channel_list
        self.wait_sec = wait_sec
