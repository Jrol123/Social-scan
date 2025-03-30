import pandas as pd
from telethon import TelegramClient


api = open('tg_api.txt', 'r').read().split('\n')

client = TelegramClient("monitoring", int(api[0]), api[1],
                        system_version="4.16.30-vxCUSTOM", app_version='1.0.1')
client.start(*api[2:4])


async def get_channel_history(channel_link, limit=100):
    data = []
    channel = (await client.get_entity(channel_link)).title
    if limit <= 1000:
        async for message in client.iter_messages(channel_link, limit=limit):
            data.append({'user': message.from_id or channel,
                         'date': message.date, 'review': message.text})
    
    return data

async def main():
    messages = await get_channel_history(
        "t.me/memoryleakage143", 20)
    df = pd.DataFrame(messages)
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['date'] = df['date'].apply(lambda x: x.timestamp())
    df.to_csv('telegram_reviews.csv', encoding='utf-8')


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
