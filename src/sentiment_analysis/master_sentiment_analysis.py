import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

AVAILABLE_LABEL_SCHEME = ["binary", "ternary"]


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
        label_scheme: str = "ternary",
        cache_dir: str | None = None,
        device: str | torch.device | None = None
    ):
        """
        Класс для семантического анализа сообщений.

        Args:
            modelPath (str): _description_
            max_length (int): _description_
            batch_size (int): _description_
            label_scheme (str, optional): Количество меток. Defaults to "ternary".
                ```
                'binary' -> 0: не-негатив, 1: негатив
                'ternary' -> 0: нейтрал, 1: позитив, 2: негатив
                ```
            cache_dir (str | None, optional): _description_. Defaults to None.
        """
        assert (
            label_scheme in AVAILABLE_LABEL_SCHEME
        ), f"Неправильная схема меток! Получено: {label_scheme}. Доступные: {AVAILABLE_LABEL_SCHEME}"
        self.DEVICE = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = batch_size
        self.MAX_LENGTH = max_length

        self.label_scheme = label_scheme

        self.tokenizer = AutoTokenizer.from_pretrained(modelPath, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelPath, cache_dir=cache_dir
        ).to(self.DEVICE)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказание label-ов

        Args:
            df (pd.DataFrame): df. Должен быть ТОЛЬКО С сообщениями без рейтинга.

        Returns:
            pd.DataFrame: _description_
        """
        texts = df["text"].apply(lambda x: str(x) if pd.notnull(x) else "").tolist()
        dataset = _PredictionDataset(texts, self.tokenizer, self.MAX_LENGTH)
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

                logits = self.__adjust_logits(
                    logits, self.model.config.num_labels, self.label_scheme
                )
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

        # TODO: Вычлинять те строки, что без рейтинга
        # TODO: Переводить рейтинг в label

        rdf = df.copy()
        rdf["label"] = predictions
        return rdf

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
