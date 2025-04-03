import requests
from bs4 import BeautifulSoup as bs


# page = requests.get('https://tgstat.ru./travels')
# page = bs(page.content, 'html.parser')
page = open('Telegram-каналы _ Россия _ Путешествия.html', 'r', encoding='utf-8')
page = bs(page.read(), 'html.parser')
channels = page.select('#category-list-form > div.row.justify-content-center.lm-list-container > div > div > a.text-body')

f = open('channel_list.txt', 'a')
for channel in channels:
    if '/@' in channel['href']:
        f.write(channel['href'].rsplit('/', 1)[1] + '\n')

f.close()
