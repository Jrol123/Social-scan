from dotenv import dotenv_values
from get_info.vk import VKParser

secrets = dotenv_values()
"""Секреты"""
vk = VKParser(secrets["VK_TOKEN"])
zp = vk.parse(count_items=20, q='Mriya Resort (Крым | Ялта) -купить', return_count=False)
zp
print(zp)