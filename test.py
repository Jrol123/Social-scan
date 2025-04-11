from dotenv import dotenv_values
from get_info.vk import VKParser

secrets = dotenv_values()
"""Секреты"""
vk = VKParser(secrets["VK_TOKEN"])
zp = vk.parse(total_count=-1, q='Mriya Resort (Крым | Ялта) -купить', return_count=True)
print(len(zp["items"]))