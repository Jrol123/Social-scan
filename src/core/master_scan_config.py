import os

from ..abstract import Config

from ..get_info.core import MasterParser, MasterParserConfig
from ..get_labels.transformers.sentiment import (
    MasterSentimentConfig,
)
from ..get_labels.transformers.rating import (
    MasterRatingConfig,
)


class MasterScanConfig(Config):
    def __init__(
        self,
        masterParser: MasterParser,
        metadata: dict[str, str],
        cache_dir: str | None,
        masterParserConfig: MasterParserConfig = None,
        masterSentimentConfig: MasterSentimentConfig = None,
        masterRatingConfig: MasterRatingConfig = None,
    ) -> None:
        super().__init__()

        self.masterParser = masterParser
        self.masterParserConfig = masterParserConfig if masterParserConfig is not None else MasterParserConfig()
        self.masterSentimentConfig = masterSentimentConfig if masterSentimentConfig is not None else MasterSentimentConfig(cache_dir=cache_dir)
        self.masterRatingConfig = masterRatingConfig if masterRatingConfig is not None else MasterRatingConfig()

        self.metadata = metadata
        self.cache_dir = cache_dir

        

        # resultT.to_csv("examples/02_example_transform.csv")
