import pandas as pd
from .config import MasterRatingConfig
from ...core import MasterTransformerConfig
from ...abstract import Transformer


class MasterRaitingTransformer(Transformer):
    def __init__(self, config: MasterRatingConfig) -> None:
        self.config = config

    def transform(self, global_config: MasterTransformerConfig) -> pd.DataFrame:
        """
        Трансформирование рейтинга в label
        """
        # 1 - Негативные отзывы

        rdf = global_config.rDf.copy()
        
        rdf["label"] = None

        for id in self.config.SERVICE_DICT.values():
            mask = rdf["service_id"] == id
            subRdf = rdf.loc[mask]
            ratings = subRdf["rating"].tolist()

            scale = self.config.service_params.get(id, None)

            if scale == self.config.default_range or scale is None:
                labels = [self.__labeler(r) for r in ratings]
            else:
                scaled_ratings = [
                    self.__scale(r, scale, self.config.default_range) 
                    for r in ratings
                ]
                labels = [self.__labeler(r) for r in scaled_ratings]

            # Используем .loc для безопасного присваивания
            rdf.loc[mask, "label"] = labels

        return rdf

    def __labeler(self, rating) -> int:
        if (rating < self.config.limit_bad) or (
            self.config.is_bad_soft and rating <= self.config.limit_bad
        ):
            # негатив
            return 1 + 1 * self.config.label_scheme

        if not self.config.label_scheme:
            # нейтраль (бинарная)
            return 0

        if (rating < self.config.limit_good) or (
            not self.config.is_good_soft and rating <= self.config.limit_good
        ):
            # нейтраль (тернарная)
            return 0

        return 1

    def __scale(
        self,
        rating: float,
        init_range: tuple[float, float],
        fin_range: tuple[float, float],
    ) -> float:
        if init_range == fin_range:
            return rating

        a, b = fin_range
        am, bm = init_range

        return a + (rating - am) / (bm - am) * (b - a)
