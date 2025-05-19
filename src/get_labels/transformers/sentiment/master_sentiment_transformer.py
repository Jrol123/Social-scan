from pandas import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader
from .config import MasterSentimentConfig
from ...core import MasterTransformerConfig
from ...abstract import Transformer


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


class MasterSentimentTransformer(Transformer):
    def __init__(self, config: MasterSentimentConfig):
        """
        Класс для семантического анализа сообщений.
        """
        self.config = config

    def transform(self, global_config: MasterTransformerConfig) -> DataFrame:
        """
        Предсказание label-ов

        Returns:
            DataFrame: _description_
        """
        sdf = global_config.sDf.copy()
        texts = sdf["text"].dropna().tolist()
        dataset = _PredictionDataset(
            texts, self.config.tokenizer, self.config.MAX_LENGTH
        )
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE)

        self.config.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.config.DEVICE),
                    "attention_mask": batch["attention_mask"].to(self.config.DEVICE),
                }
                outputs = self.config.model(**inputs)
                logits = outputs.logits

                logits = self.__adjust_logits(
                    logits,
                    self.config.model.config.num_labels,
                    self.config.label_scheme,
                )
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

        sdf["label"] = predictions
        return sdf

    @staticmethod
    def __adjust_logits(logits, num_labels, label_scheme):
        """
        Параметры:
        label_scheme:
            'binary' -> 0: не-негатив, 1: негатив
            'ternary' -> 0: нейтрал, 1: позитив, 2: негатив
        """
        assert num_labels in [
            3,
            5,
        ], f"Неподдерживаемое число меток. Полученное: {num_labels} Доступное: {[3, 5]}"
        if num_labels == 3:
            if label_scheme == "binary":
                # Объединяем нейтрал (0) и позитив (1) в класс 0
                # Негатив (2) остается классом 1
                neg_logits = logits[:, 2]
                other_logits = torch.max(logits[:, :2], dim=1)[0]
                return torch.stack([other_logits, neg_logits], dim=1)

            elif label_scheme == "ternary":
                # Оставляем оригинальные классы
                return logits

        elif num_labels == 5:
            # Классы: 0:negative, 1:neutral, 2:positive, 3:skip, 4:speech
            if label_scheme == "binary":
                # 0: нейтрал (1) + позитив (2) + skip (3) + speech (4)
                # 1: негатив (0)
                non_negative = torch.max(logits[:, [1, 2, 3, 4]], dim=1)[0]
                return torch.stack([non_negative, logits[:, 0]], dim=1)

            elif label_scheme == "ternary":
                # 0: нейтрал (max из 1,3,4)
                # 1: позитив (2)
                # 2: негатив (0)
                neutral = torch.max(logits[:, [1, 3, 4]], dim=1)[0]
                positive = logits[:, 2]
                negative = logits[:, 0]
                return torch.stack([neutral, positive, negative], dim=1)

        return logits


# "sismetanin/mbart_ru_sum_gazeta-ru-sentiment-rusentiment"
