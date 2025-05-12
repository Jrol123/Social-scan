import pandas as pd
from .yandex_reviews_parser.utils import YandexMapsParser

id_ya = 1303073708  # ID Компании Yandex
parser = YandexMapsParser()

all_data = parser.parse(id_ya, count_items=10)
print(all_data)
# all_data

df = pd.DataFrame(all_data).rename(columns={'start': 'rating'})

# df = pd.DataFrame(all_data['company_reviews'])
# df.to_csv("lol.csv")
# print(all_data)
