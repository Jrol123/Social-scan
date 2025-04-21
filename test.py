from dotenv import dotenv_values
from get_info.vk import VKParser

from datetime import datetime

secrets = dotenv_values()
"""Секреты"""
vk = VKParser(secrets["VK_TOKEN"])
zp = vk.parse(count_items=9999999, q='Mriya Resort (Крым | Ялта) -купить', min_date=datetime(2024,1,1), return_count=False)
zp
print(zp)