from dotenv import dotenv_values
from vk import VKParser

secrets = dotenv_values()
"""Секреты"""
vk = VKParser(secrets["VK_TOKEN"])
zp = vk.search_feed(total_count=-1, q='Mriya Resort (Крым | Ялта) -купить')
print(len(zp["items"]))
"""
МРИЯ / 342
МРИЯ ("Отель у моря") (Крым | Ялта) -купить / 106
Мрия (отель | курорт | санаторий | гостиница) (Крым | Ялта) -купить / 4
Отель Мрия / 3275
Отель МРИЯ (Крым | Ялта) / 1343
Отель МРИЯ (Крым | Ялта) -купить / 609
Мрия курорт / 2114
Санаторий МРИЯ / 1225
МРИЯ гостиница / 1071
МРИЯ РЕЗОРТ энд СПА / 701
Mriya Resort / 7759
Mriya Resort (Крым | Ялта) -купить / 1463
Mriya Resort&Spa / 6608
Mriya Resort&Spa (Крым | Ялта) / 1354
"""
