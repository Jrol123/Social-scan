from dotenv import dotenv_values
from .vk import VKParser, VKConfig
from ...core import MasterParserConfig


if __name__ == "__main__":
    secrets = dotenv_values()
    # vk
    global_config = MasterParserConfig()
    texts = [
        "МРИЯ",
        "МРИЯ ('Отель у моря') (Крым | Ялта) -купить",
        "Мрия (отель | курорт | санаторий | гостиница) (Крым | Ялта) -купить",
        "Отель Мрия",
        "Отель МРИЯ (Крым | Ялта)",
        "Отель МРИЯ (Крым | Ялта) -купить",
        "Мрия курорт",
        "Санаторий МРИЯ",
        "МРИЯ гостиница",
        "МРИЯ РЕЗОРТ энд СПА",
        "Mriya Resort",
        "Mriya Resort (Крым | Ялта) -купить",
        "Mriya Resort&Spa",
        "Mriya Resort&Spa (Крым | Ялта)",
    ]
    configs = [VKConfig(q = text, return_only_count=True) for text in texts]
    parsers = [VKParser(secrets["VK_TOKEN"], config) for config in configs]
    for text, parser in zip(texts, parsers):
        result = parser.parse(global_config=global_config)
        print(text, result, sep=" / ")