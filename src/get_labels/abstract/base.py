from abc import ABC, abstractmethod
from pandas import DataFrame
from ...abstract import Config


class Transformer(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def transform(self, global_config) -> DataFrame:
        """
        --

        Args:
            global_config (_type_): _description_

        Returns:
            DataFrame: _description_
        """
        pass
