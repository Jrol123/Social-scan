"""
Этот пример посвящён парсингу, первому этапу в pipeline.

Пример
"""

from datetime import datetime

import pandas as pd
from dotenv import dotenv_values

from src.get_info.parsers import (
    VKParser,
    TelegramParser,
    GoogleMapsParser,
    YandexMapsParser,
    OtzovikParser,
)
from src.get_info.core import MasterParser

secrets = dotenv_values()
"""Секреты"""

min_date = datetime(year=2024, month=1, day=1)
mPars = MasterParser(
    # TelegramParser(secrets["TG_ID"], secrets["TG_HASH"], secrets["PHONE"],
    #                secrets["PASSWORD"]),
    VKParser(secrets["VK_TOKEN"]),
    YandexMapsParser(),
    GoogleMapsParser(),
    OtzovikParser(),
    VKParser={
        "q": "Mriya Resort (Крым | Ялта) -купить",
    },
    # TelegramParser={"q": "МРИЯ",
    #                 "channels_list": ["t.me/mriyaresortyalta","t.me/mriyaresortchat"],
    #                 "limit": 1000,
    #                 "min_date": min_date},
    YandexMapsParser={"q": 1303073708},
    GoogleMapsParser={
        "q": r"https://www.google.com/maps/place/?q=place_id:ChIJ7WjSWynClEARUUiva4PiDzI",
        "collect_extra": False,
        "wait_load": 30,
    },
    OtzovikParser={
        "q": r"https://otzovik.com/reviews/sanatoriy_mriya_resort_spa_russia_yalta/",
    },
)
data = mPars.parse(count_items=-1, min_date=min_date)

tgparser = TelegramParser(
    secrets["TG_ID"], secrets["TG_HASH"], secrets["PHONE"], secrets["PASSWORD"]
)
with tgparser.client:
    result = tgparser.client.loop.run_until_complete(
        tgparser.parse(
            **{
                "q": "МРИЯ",
                "channels_list": [
                    "t.me/mriyaresortchat"
                ],  # , "t.me/mriyaresortyalta"],
                "limit": 1000,
                "min_date": min_date,
            }
        )
    )

data.extend(result)

# loop = asyncio.get_event_loop()
# df = pd.DataFrame(asyncio.run(main()))
df = pd.DataFrame(data)
df.to_csv("parsed_data.csv")
