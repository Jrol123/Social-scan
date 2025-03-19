import pandas as pd
from yandex_reviews_parser.utils import YandexParser
id_ya = 66936682838 #ID Компании Yandex
parser = YandexParser(id_ya)

all_data = parser.parse() #Получаем все данные
df = pd.DataFrame(all_data)
df.to_csv("lol.csv")
print(all_data)