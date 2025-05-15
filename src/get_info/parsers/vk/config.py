from ...abstract import ParserConfig
from dataclasses import dataclass


@dataclass
class VKConfig(ParserConfig):
    q: str
    return_count: bool
    fields: str

    def __init__(
        self,
        q: str,
        return_only_count: bool = False,
        fields: str = "id, first_name, last_name",
        start_from: str = "0",
    ) -> None:
        """
        Конфиг для VKParser

        Args:
            q (str): Текст запроса. Используются регулярные выражения.
            return_only_count (bool, optional): Возвращать ли только количество результатов, или нет. Defaults to False.
            fields (str, optional): Поля, которые нужно возвращать. Defaults to "id, first_name, last_name".
            start_from (str, optional): Локальный id начального сообщения (сдвиг). Defaults to "0".

        """
        self.q = q
        self.return_only_count = return_only_count
        self.fields = fields
        self.start_from = start_from
