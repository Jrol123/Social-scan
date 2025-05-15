"""
Этот пример посвящён парсингу, первому этапу в pipeline.

Пример
"""

import asyncio

from dotenv import dotenv_values

from src.get_info.core import MasterParser, MasterConfig
from src.get_info.parsers.google_maps import GoogleMapsConfig, GoogleMapsParser
from src.get_info.parsers.otzovik import OtzovikConfig, OtzovikParser
from src.get_info.parsers.telegram import TelegramConfig, TelegramParser
from src.get_info.parsers.vk import VKParser, VKConfig
from src.get_info.parsers.yandex_maps import YandexMapsParser, YandexMapsConfig


secrets = dotenv_values()


async def main():
    global_config = MasterConfig(count_items=10, sort_type="Сначала положительные")

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

    # Инициализируем клиент Telegram внутри общего event loop
    async with tg_parser.client:
        master_parser = MasterParser(
            google_parser, otzovik_parser, yandex_parser, vk_parser, tg_parser
        )
        results = await master_parser.async_parse(global_config)
        print(results)


if __name__ == "__main__":
    # Используем явное создание event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
