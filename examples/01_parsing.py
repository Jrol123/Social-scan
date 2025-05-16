"""
Этот пример посвящён парсингу, первому этапу в pipeline.
"""

import asyncio

from dotenv import dotenv_values
from pandas import DataFrame

from src.get_info.core import MasterParser, MasterParserConfig
from src.get_info.parsers.google_maps import GoogleMapsConfig, GoogleMapsParser
from src.get_info.parsers.otzovik import OtzovikConfig, OtzovikParser
from src.get_info.parsers.telegram import TelegramConfig, TelegramParser
from src.get_info.parsers.vk import VKParser, VKConfig
from src.get_info.parsers.yandex_maps import YandexMapsParser, YandexMapsConfig


secrets = dotenv_values()


# Асинхронная функция нужна для работы асинхронных парсеров
async def main():
    # Подготовка конфигураций
    
    global_config = MasterParserConfig(
        count_items=100, sort_type="Сначала отрицательные"
    )

    # В парсер идёт его конфигурация + необходимые параметры (такие как токен для vk)
    google_config = GoogleMapsConfig(
        r"https://www.google.com/maps/place/?q=place_id:ChIJ7WjSWynClEARUUiva4PiDzI"
    )
    google_parser = GoogleMapsParser(google_config)

    yandex_config = YandexMapsConfig(1303073708)
    yandex_parser = YandexMapsParser(yandex_config)

    otzovik_config = OtzovikConfig(
        "https://otzovik.com/reviews/sanatoriy_mriya_resort_spa_russia_yalta/"
    )
    otzovik_parser = OtzovikParser(otzovik_config)

    vk_config = VKConfig(q="Mriya Resort (Крым | Ялта) -купить")
    vk_parser = VKParser(secrets["VK_TOKEN"], vk_config)

    tg_config = TelegramConfig("МРИЯ", ["t.me/mriyaresortchat"])
    tg_parser = TelegramParser(
        tg_config,
        int(secrets["TG_ID"]),
        secrets["TG_HASH"],
        secrets["PHONE"],
        secrets.get("PASSWORD"),
    )

    # Парсинг
    master_parser = MasterParser(
        tg_parser, otzovik_parser, yandex_parser, vk_parser, google_parser
    )
    results = await master_parser.async_parse(global_config)

    return results


if __name__ == "__main__":
    # Используем явное создание event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = DataFrame(loop.run_until_complete(main()))
    finally:
        loop.close()

    result.to_csv("test_parse.csv")
