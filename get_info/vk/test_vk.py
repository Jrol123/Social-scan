from dotenv import dotenv_values
from vk import VKParser

secrets = dotenv_values(".env")
"""Секреты"""
vk = VKParser(secrets["VK_TOKEN"])
zp = vk.search_feed(total_count=-1, q="мрия плохо")
print(len(zp["items"]))
