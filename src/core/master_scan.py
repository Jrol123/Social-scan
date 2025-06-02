import os
from pandas import DataFrame, read_csv

from ..get_labels.core import MasterTransformer, MasterTransformerConfig
from ..get_labels.transformers.sentiment import (
    MasterSentimentTransformer,
)
from ..get_labels.transformers.rating import (
    MasterRaitingTransformer,
)
from ..get_summarization import (
    gen_categories,
    gen_summarization,
    gen_multilabel_summarization,
)
from ..get_clusterization import MasterClusterization
from ..get_report import form_report

from .master_scan_config import MasterScanConfig


class MasterScan:
    def __init__(self, config: MasterScanConfig) -> None:
        self.config = config

    def generate_report(
        self,
        output_name: str,
        mistralKey: str,
        deepseekKey: str,
        is_multilabel: bool = True,
        multilabelClasses: list[str] = None,
    ):
        print("GET DATA")
        data = self.config.masterParser.sync_parse(self.config.masterParserConfig)
        data = DataFrame(
            data,
            # dtype={
            #     "service_id": "int32",
            #     "date": "int64",
            #     "rating": "float32",
            #     "name": "object",
            #     "additional_id": "object",
            #     "text": "object",
            #     "answer": "object",
            #     # "label": "int32",
            # },
        )

        data = data.astype(
            dtype={
                "service_id": "int32",
                "date": "int64",
                "rating": "float32",
                "name": "object",
                "additional_id": "object",
                "text": "object",
                "answer": "object",
                # "label": "int32",
            }
        )

        # for column in data.columns:
        #     data[column] = data[column].astype(dtype[column])

        # dtype=[
        #         ("service_id", "int32"),
        #         ("date", "int64"),
        #         ("rating", "float32"),
        #         ("name", "object"),
        #         ("additional_id", "object"),
        #         ("text", "object"),
        #         ("answer", "object"),
        #         ("label", "int32"),
        #     ],

        data = data.dropna(how="all")  # На всякий случай
        data = data[~data["text"].isna()]

        print()

        print("TRANSFORM")
        ratT = MasterRaitingTransformer(self.config.masterRatingConfig)
        senT = MasterSentimentTransformer(self.config.masterSentimentConfig)

        mtf = MasterTransformerConfig(data)
        mts = MasterTransformer(mtf)
        result = mts.transform(ratT, senT)

        print()

        print("SUMMARIZATION")
        is_ternary = self.config.masterSentimentConfig.label_scheme == "ternary"

        df = result[result["label"] == 1 + is_ternary]

        if is_multilabel:
            multilabelClasses = (
                gen_categories(df, mistralKey, "mistral", self.config.metadata)
                if multilabelClasses is None
                else multilabelClasses
            )

            summaries = gen_multilabel_summarization(
                df,
                multilabelClasses,
                metadata=self.config.metadata,
                token=deepseekKey,
                model_name="deepseek",
            )

        else:
            summaries = gen_summarization(
                df,
                token=mistralKey,
                model_name="mistral",
            )

        df["summary"] = summaries

        print()

        print("CLUSTERIZATION")
        TMP_FOLDER = "SOCIAL_SCAN_TMP"

        os.mkdir(TMP_FOLDER)

        MasterClusterization(
            df, deepseekKey, 100, TMP_FOLDER, cache_dir=self.config.cache_dir
        )

        summaries = read_csv(
            os.path.join(TMP_FOLDER, "clustered_summaries2.csv"), index_col=0
        )
        clusters = read_csv(os.path.join(TMP_FOLDER, "categories.csv"), index_col=0)

        print()

        print("REPORT")
        form_report(summaries, clusters, deepseekKey, self.config.metadata, output_name)

        os.rmdir(TMP_FOLDER)
