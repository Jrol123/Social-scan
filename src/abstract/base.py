from abc import ABC, abstractmethod
from datetime import datetime
from typing import Type


class Config(ABC):
    def _check_data(self, time: datetime | int | None) -> datetime | int:
        if time is None:
            return datetime.now()
        return time


class GlobalConfig(Config):
    """
    Класс, содержащий общие параметры.
    """
