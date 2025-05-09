import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification


class __PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            max_length,
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
    def __init__(self, modelPath: str, max_length: int, batch_size: int):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = batch_size
        self.MAX_LENGTH = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelPath).to(
            self.DEVICE
        )

    def predict(self, df: pd.DataFrame):
        dataset = __PredictionDataset(
            df["text"].tolist(), self.tokenizer, self.MAX_LENGTH
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
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        rdf = df.copy()
        rdf["semLabel"] = predictions
        return rdf


# "sismetanin/mbart_ru_sum_gazeta-ru-sentiment-rusentiment"
