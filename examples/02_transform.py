"""
Этот пример посвящён разметке, второму этапу в pipeline.

Здесь мы определяем, в какую из двух (трёх) групп попадёт тот или иной отзыв.
"""
from pandas import read_csv

from src.sentiment_analysis.core import MasterTransformer, MasterTransformerConfig
from src.sentiment_analysis.transformers.sentiment import (
    MasterSentimentTransformer,
    MasterSentimentConfig,
)
from src.sentiment_analysis.transformers.rating import (
    MasterRaitingTransformer,
    MasterRatingConfig,
)

if __name__ == "__main__":
    results = read_csv(
        "test_parse.csv",
        index_col=0,
        dtype={
            "service_id": "int32",
            "date": "int64",
            "rating": "float32",
            "name": "object",
            "additiona_id": "object",
            "text": "object",
            "answer": "object",
            "label": "int32",
        },
    )

    ratC = MasterRatingConfig(limit_bad=2.0, is_bad_soft=False, is_good_soft=True)
    ratT = MasterRaitingTransformer(ratC)

    senC = MasterSentimentConfig(
        modelPath="sismetanin/mbart_ru_sum_gazeta-ru-sentiment-rusentiment",
        batch_size=12,
        cache_dir="D:/TRANSFORMERS_MODELS",
    )
    senT = MasterSentimentTransformer(senC)

    mtf = MasterTransformerConfig(results)
    mts = MasterTransformer(mtf)
    resultT = mts.transform(ratT, senT)
    
    resultT.to_csv("test_transform.csv")
