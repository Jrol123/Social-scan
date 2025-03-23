import pandas as pd
from yandex_reviews_parser.utils import YandexParser
id_ya = 1303073708 #ID Компании Yandex
parser = YandexParser(id_ya)

while True:
    # TODO: Почему-то периодически вылезает "неудалось кликнуть"
    all_data = parser.parse("company") #Получаем все данные
    if all_data.get('company_info'):
        break
print(all_data)
# all_data

# df = pd.DataFrame(all_data['company_reviews'])
# df.to_csv("lol.csv")
# print(all_data)
