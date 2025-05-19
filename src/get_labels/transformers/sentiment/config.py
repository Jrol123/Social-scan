import torch
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from ....abstract import Config

AVAILABLE_LABEL_SCHEME = ["binary", "ternary"]
"""Возможные схемы разметки"""


class MasterSentimentConfig(Config):
    def __init__(
        self,
        modelPath: str,
        batch_size: int,
        label_scheme: str = "ternary",
        cache_dir: str | None = None,
    ) -> None:
        """
        Конфиг для семантики.

        Args:
            modelPath (str): Путь до модели. Модель должна быть как на `huggingface.co`
            max_length (int): _description_
            batch_size (int): _description_
            label_scheme (str, optional): Схема разметки. Defaults to "ternary".
                ```
                'binary' -> 0: не-негатив, 1: негатив
                'ternary' -> 0: нейтрал, 1: позитив, 2: негатив
                ```
            cache_dir (str | None, optional): Путь до модели на локальном компьютере. При None будет указана стандартная директория для `huggingface.co`. Defaults to None.
        """
        super().__init__()
        assert (
            label_scheme in AVAILABLE_LABEL_SCHEME
        ), f"Неправильная схема меток! Получено: {label_scheme}. Доступные: {AVAILABLE_LABEL_SCHEME}"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = batch_size

        self.label_scheme = label_scheme
        """
        Схема разметки.
        
        ```
        'binary' -> 0: не-негатив, 1: негатив
        'ternary' -> 0: нейтрал, 1: позитив, 2: негатив
        ```
        """

        self.tokenizer = AutoTokenizer.from_pretrained(modelPath, cache_dir=cache_dir)
        """Токенизатор"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelPath, cache_dir=cache_dir
        ).to(self.DEVICE)
        """Модель"""
        
        try:
            self.MAX_LENGTH = self.model.config.max_position_embeddings
        except AttributeError:
            self.MAX_LENGTH = (self.tokenizer.model_max_length
                    if self.tokenizer.model_max_length <= 2**20 else 512)
