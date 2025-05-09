import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification


class _PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }

    def __len__(self):
        return len(self.encodings["input_ids"])


class MasterSentimentAnalysis:
    def __init__(
        self,
        modelPath: str,
        max_length: int,
        batch_size: int,
        cache_dir: str | None = None,
    ):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = batch_size
        self.MAX_LENGTH = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(modelPath, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelPath, cache_dir=cache_dir
        ).to(self.DEVICE)

    def predict(self, df: pd.DataFrame):
        df["text"] = df["text"].astype(str)
        df = df.dropna(subset=["text"])
        texts = df["text"].apply(lambda x: str(x) if pd.notnull(x) else "").tolist()
        dataset = _PredictionDataset(
            texts, self.tokenizer, self.MAX_LENGTH
        )
        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.DEVICE),
                    "attention_mask": batch["attention_mask"].to(self.DEVICE),
                }
                outputs = self.model(**inputs)
                logits = outputs.logits
                if self.model.config.num_labels == 3:
                    logits[:, 1] = torch.max(logits[:, :2], dim=1)[0]
                    logits = logits[:, 1:]
                elif self.model.config.num_labels == 5:
                    logits[:, 1] = torch.max(logits[:, 1:], dim=1)[0]
                    logits = torch.column_stack([logits[:, 1], logits[:, 0]])
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                
        # TODO: Вычлинять те строки, что без рейтинга
        # TODO: Переводить рейтинг в label
                
        rdf = df.copy()
        rdf["semLabel"] = predictions
        return rdf


# "sismetanin/mbart_ru_sum_gazeta-ru-sentiment-rusentiment"
