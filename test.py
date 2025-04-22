from dotenv import dotenv_values
from get_info.vk import VKParser

import pandas as pd
from datetime import datetime

secrets = dotenv_values()
"""Секреты"""
vk = VKParser(secrets["VK_TOKEN"])

list_requests = [
    # "МРИЯ -купить -вакансия",
# "МРИЯ ('Отель у моря') (Крым | Ялта) -купить -вакансия",
# "Мрия (отель | курорт | санаторий | гостиница) (Крым | Ялта) -купить -вакансия"
# "Отель Мрия -купить -вакансия",
# "Отель МРИЯ (Крым | Ялта) -купить -вакансия",
# Мрия курорт -купить -вакансия
# Санаторий МРИЯ -купить -вакансия
# МРИЯ гостиница -купить -вакансия
# МРИЯ РЕЗОРТ энд СПА -купить -вакансия
# Mriya Resort -купить -вакансия
"Mriya Resort (Крым | Ялта) -купить -вакансия",
# "Mriya Resort&Spa -купить -вакансия",
# "Mriya Resort&Spa (Крым | Ялта) -купить -вакансия"
]

iter = 0

zp = []

for iter, q in enumerate(list_requests):
    print(q)
    # q = file.readline()
    # q = "Mriya Resort (Крым | Ялта) -купить -вакансия"
    zp.extend(vk.parse(count_items=-1, q=q, min_date=datetime(2024,1,1), return_count=False))
    # TODO: придумать, как правильно объединять сообщения. НЕ ПРИОРИТЕТ
df = pd.DataFrame(data=zp)
df.to_csv(path_or_buf=f"sentiment_analysis/vk/vk_messages_mriya.csv")