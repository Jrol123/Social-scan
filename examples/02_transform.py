"""
Этот пример посвящён разметке, второму этапу в pipeline.

Здесь определяется, в какую из двух (трёх) групп попадёт тот или иной отзыв.
"""
from pandas import read_csv

from src.get_labels.core import MasterTransformer, MasterTransformerConfig
from src.get_labels.transformers.sentiment import (
    MasterSentimentTransformer,
    MasterSentimentConfig,
)
from src.get_labels.transformers.rating import (
    MasterRaitingTransformer,
    MasterRatingConfig,
)

if __name__ == "__main__":
    results = read_csv(
        "examples/01_example_parse.csv",
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
    results = results.dropna(how='all')
    results = results[~results['text'].isna()]
    
    ratC = MasterRatingConfig(limit_bad=3.0, is_bad_soft=True, is_good_soft=False)
    ratT = MasterRaitingTransformer(ratC)
    
    # sismetanin/sbert-ru-sentiment-rusentiment
    # sismetanin/mbart_ru_sum_gazeta-ru-sentiment-rusentiment
    senC = MasterSentimentConfig(
        modelPath="seara/rubert-tiny2-russian-sentiment",
        batch_size=12,
        # cache_dir="D:/TRANSFORMERS_MODELS",
        device="cpu"
    )
    senT = MasterSentimentTransformer(senC)
    
    mtf = MasterTransformerConfig(results)
    mts = MasterTransformer(mtf)
    resultT = mts.transform(ratT, senT)
    
    resultT.to_csv("examples/02_example_transform.csv")
