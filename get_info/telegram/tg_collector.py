import time
from datetime import datetime

import pandas as pd
from telethon import TelegramClient


api = open('tg_api.txt', 'r').read().split('\n')

client = TelegramClient("monitoring", int(api[0]), api[1],
                        system_version="4.16.30-vxCUSTOM", app_version='1.0.1')
client.start(*api[2:4])


async def get_channel_history(channel_link, limit=100):
    data = []
    channel = await client.get_entity(channel_link)
    channel_id = channel.id
    channel_name = channel.title
    if limit is None:
        channel_history = [message async for message in client.iter_messages(
                           channel_link, limit=1000)]
        while channel_history:
            for message in channel_history:
                if message.text:
                    user = client.get_entity(message.from_id)
                    data.append({'user_id': message.from_id or channel_id,
                                 'username': user.username or channel_name,
                                 'user': user.first_name + ' ' + user.last_name
                                         or channel_name,
                                 'date': message.date, 'review': message.text})
            
            last_date = data[-1]['date']
            channel_history = [message async for message in client.iter_messages(
                               channel_link, limit=1000, offset_date=last_date)]
                
    elif limit <= 1000:
        async for message in client.iter_messages(channel_link, limit=limit):
            if message.text:
                data.append({'user': message.from_id or channel,
                             'date': message.date, 'review': message.text})
    else:
        for i, offset in enumerate(range(0, limit, 1000)):
            async for message in client.iter_messages(
               channel_link, limit=1000 if i > limit // 1000 else i % 1000):
                if message.text:
                    data.append({'user': message.from_id or channel,
                                 'date': message.date, 'review': message.text})
    
    return data

async def parse_all_channels(channels_list="channel_list.txt", limit=100):
    if isinstance(channels_list, str):
        channels_list = open(channels_list).readlines()
        channels_list = list(map(str.strip, channels_list))
    
    data = []
    for channel in channels_list:
        data.extend(await get_channel_history(channel, limit))
        time.sleep(0.5)
    
    return data

def save_reviews_to_csv(reviews, min_date=None, filename="telegram_reviews.csv"):
    df = pd.DataFrame(reviews)
    df['date'] = pd.to_datetime(df['date'], unit='s')
    if isinstance(min_date, str):
        df = df[df['date'] > datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S")]
    elif isinstance(min_date, datetime):
        df = df[df['date'] > min_date]
    
    df['date'] = df['date'].apply(lambda x: x.timestamp())
    df = df.sort_values('date', ascending=False)
    df.to_csv(filename, encoding='utf-8')

async def telegram_parse(channels_list="channel_list.txt", limit=100,
                         min_date=None, filename="telegram_reviews.csv"):
    messages = await parse_all_channels(channels_list, limit)
    save_reviews_to_csv(messages, min_date, filename)
    

async def main():
    await telegram_parse()


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
