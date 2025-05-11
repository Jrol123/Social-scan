import requests
from bs4 import BeautifulSoup as bs


# page = requests.get('https://tgstat.ru./travels')
# page = bs(page.content, 'html.parser')
page = open('Telegram-чаты _ Россия _ Путешествия.html', 'r', encoding='utf-8')
page = bs(page.read(), 'html.parser')
channels = page.select('#category-list-form > div.row.justify-content-center.lm-list-container > div > div > a.text-body')

channel_list = open('channel_list.txt', 'r').readlines()
channel_list = [c.strip() for c in channel_list]
for channel in channels:
    if '/@' in channel['href']:
        channel = channel['href'].rsplit('/', 1)[1]
    else:
        continue
    
    if channel not in channel_list:
        channel_list.append(channel)

with open('channel_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(channel_list))
