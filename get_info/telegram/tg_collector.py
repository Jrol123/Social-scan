import os
import time
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from telethon import TelegramClient


load_dotenv()

api_id = os.environ.get("TG_ID")
api_hash = os.environ.get("TG_HASH")
phone_number = os.environ.get("PHONE")
password = os.environ.get("PASSWORD")
# api = open('tg_api.txt', 'r').read().split('\n')

client = TelegramClient("monitoring", int(api_id), api_hash,
                        system_version="4.16.30-vxCUSTOM", app_version='1.0.1')
client.start(phone_number, password)


async def form_line(message, client, channel_id, channel_name):
    try:
        user = await client.get_entity(message.from_id)
        user_id = user.id
        username = user.username
        user = user.first_name + ' ' + user.last_name
    except (TypeError, ValueError):
        username = None
        user = None
        user_id = None
    
    return {'user_id': user_id or channel_id,
            'username': username or channel_name,
            'user': user if user is not None and user != ' ' else channel_name,
            'date': message.date.replace(tzinfo=None),
            'review': message.text}

async def get_channel_history(channel_link, limit=100, search=None, min_date=None):
    data = []
    try:
        channel = await client.get_entity(channel_link)
    except ValueError:
        return []
    
    channel_id = channel.id
    channel_name = channel.title
    print(channel_link, end=' ')
    if limit is None:
        channel_history = [message async for message in client.iter_messages(
                           channel_link, limit=100, search=search)]
        while channel_history:
            for message in channel_history:
                if message.text:
                    data.append(await form_line(message, client,
                                                channel_id, channel_name))
            
            last_date = data[-1]['date']
            channel_history = [message async for message in client.iter_messages(
                               channel_link, limit=100,
                               offset_date=last_date, search=search)]
            if min_date:
                channel_history = list(filter(
                    lambda m: m.date.replace(tzinfo=None) > min_date,
                    channel_history))
            
    elif limit <= 100:
        async for message in client.iter_messages(channel_link, limit=limit,
                                                  search=search):
            if message.text:
                data.append(await form_line(message, client,
                                            channel_id, channel_name))
    else:
        async for message in client.iter_messages(channel_link, limit=100,
                                                  search=search):
            if message.text:
                data.append(await form_line(message, client,
                                            channel_id, channel_name))
        
        offset_data = None
        for i in range(100, limit, 100):
            try:
                offset_data = data[-1]['date']
            except IndexError:
                pass
            
            async for message in client.iter_messages(
               channel_link, limit=100 if i + 100 < limit
               else abs(limit - i*100) % 100,
               offset_date=offset_data, search=search):
                if message.text:
                    data.append(await form_line(message, client,
                                                channel_id, channel_name))
                
            if min_date and data and data[-1]['date'] < min_date:
                break
    
    print(len(data))
    return data

async def parse_all_channels(channels_list="channel_list.txt", limit=100, search=None, min_date=None):
    if isinstance(channels_list, str):
        channels_list = open(channels_list).readlines()
        channels_list = list(map(str.strip, channels_list))
    
    data = []
    for channel in channels_list:
        data.extend(await get_channel_history(channel, limit, search, min_date))
        time.sleep(0.5)
    
    return data

def save_reviews_to_csv(reviews, min_date=None, filename="telegram_reviews.csv"):
    if not reviews:
        print('There is no data collected from telegram.')
        return
    
    df = pd.DataFrame(reviews)
    df['date'] = pd.to_datetime(df['date'], unit='s')
    if isinstance(min_date, str):
        df = df[df['date'] > datetime.strptime(min_date, "%Y-%m-%d")]
    elif isinstance(min_date, datetime):
        df = df[df['date'] > min_date]
    
    if df is None or df.empty:
        print('There is no data collected from telegram.')
        return
    
    df['date'] = df['date'].apply(lambda x: x.timestamp())
    df = df.sort_values('date', ascending=False).reset_index(drop=True)
    df.to_csv(filename, index=False, encoding='utf-8')

async def telegram_parse(channels_list="channel_list.txt", search=None, limit=100,
                         min_date=None, filename="telegram_reviews.csv"):
    messages = await parse_all_channels(channels_list, limit, search, min_date)
    save_reviews_to_csv(messages, min_date, filename)
    

async def main():
    await telegram_parse(limit=500,
                         search='МРИЯ гостиница',
                         min_date=datetime(year=2025, month=2, day=7),
                         filename='mriya_messages.csv')


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
    """
    МРИЯ / 6
    Отель Мрия / 1
    Мрия курорт / 4
    """
