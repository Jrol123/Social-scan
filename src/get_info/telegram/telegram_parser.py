import os
import time
from datetime import datetime

import pandas as pd
from Demos.mmapfile_demo import offset
from dotenv import load_dotenv
from telethon import TelegramClient

from src.get_info.abstract import Parser


class TelegramParser(Parser):
    def __init__(self, api_id, api_hash, phone, password=None,
                 session_name="monitoring", system_version="4.16.30-vxCUSTOM",
                 app_version="1.0.1"):
        super().__init__(4)
        
        self.client = TelegramClient(session_name, api_id, api_hash,
                                     system_version=system_version,
                                     app_version=app_version)
        self.client.start(phone, password)
    
    async def parse(
        self,
        q: str | list[str],
        channels_list="channel_list.txt",
        min_date: datetime | int = datetime(1970, 1, 16),
        max_date: datetime | int = datetime.now(),
        count_items: int = -1,
        wait_sec=1
    ) -> list[dict[str, str | int | float | None]]:
        
        if isinstance(channels_list, str):
            channels_list = open(channels_list).readlines()
            channels_list = list(map(str.strip, channels_list))
            
        data = []
        for channel in channels_list:
            data.extend(await self.get_channel_history(
                channel, q, count_items, min_date, max_date, wait_sec))
            time.sleep(wait_sec)
        
        if not data:
            print('There is no data collected from telegram.')
            return []
        
        return data

    async def get_channel_history(
        self,
        channel_link,
        q,
        count_items: int,
        min_date: datetime | int,
        max_date: datetime | int,
        wait_sec
    ):
        data = []
        try:
            channel = await self.client.get_entity(channel_link)
        except ValueError:
            return []
        
        channel_id = channel.id
        print(channel_link)
        if count_items is None:
            channel_history = [
                message async for message in self.client.iter_messages(
                    channel_link, limit=100, search=q, offset_date=max_date)
            ]
            while channel_history:
                time.sleep(wait_sec)
                for message in channel_history:
                    if message.text:
                        data.append(await self.__form_line(message, channel_id))
                
                last_date = data[-1]['date']
                channel_history = [
                    message async for message in self.client.iter_messages(
                        channel_link, limit=100, offset_date=last_date, search=q)
                ]
                if min_date is not None:
                    channel_history = list(filter(
                        lambda m: m.date.replace(tzinfo=None) > min_date,
                        channel_history))
                
        elif count_items <= 100:
            async for message in self.client.iter_messages(
                  channel_link, limit=count_items, search=q, offset_date=max_date):
                if message.text:
                    data.append(await self.__form_line(message, channel_id))
        else:
            async for message in self.client.iter_messages(
                  channel_link, limit=100, search=q, offset_date=max_date):
                if message.text:
                    data.append(await self.__form_line(message, channel_id))
            
            offset_data = None
            for i in range(100, count_items, 100):
                time.sleep(1)
                try:
                    offset_data = data[-1]['date']
                except IndexError:
                    pass
                
                async for message in self.client.iter_messages(
                    channel_link,
                    limit=100 if i + 100 < count_items
                          else (i - count_items) % 100,
                    offset_date=offset_data,
                    search=q
                ):
                    if message.text:
                        data.append(await self.__form_line(message, channel_id))
                    
                if min_date is not None and data and data[-1]['date'] < min_date:
                    break
        
        if min_date is not None:
            dellines = []
            for i in range(len(data)):
                if data[i]['date'] < min_date:
                    dellines.append(i)
                
                data[i]['date'] = data[i]['date'].timestamp()
            
            if dellines:
                data = [data[i] for i in range(len(data)) if i not in dellines]
        else:
            data = [data[i]['date'].timestamp() for i in range(len(data))]
        
        # print(len(data))
        return data
    
    async def __form_line(self, message, channel_id):
        try:
            # TODO: оптимизировать получение объекта пользователя путём кэширования
            user = await self.client.get_entity(message.from_id)
            user_id = user.id
        except (TypeError, ValueError, AttributeError):
            user_id = None
        
        return {'service_id': self.service_id,
                'name': user_id or channel_id,
                'additional_id': channel_id if user_id else None,
                'date': message.date.replace(tzinfo=None),
                'rating': None,
                'text': message.text,
                'answer': None}


def save_reviews_to_csv(reviews, filename="telegram_reviews.csv"):
    if not reviews:
        return
    
    df = pd.DataFrame(reviews)
    df = df.sort_values('date', ascending=False).reset_index(drop=True)
    df.to_csv(filename, index=False, encoding='utf-8')

async def telegram_parse(parser: TelegramParser, channels_list="channel_list.txt",
                         search=None, count_items=100, min_date=None, max_date=None,
                         filename="telegram_reviews.csv"):
    messages = await parser.parse(search, channels_list,
                                  min_date, max_date, count_items)
    save_reviews_to_csv(messages, filename)
    

def main():
    load_dotenv()
    
    api_id = os.environ.get("TG_ID")
    api_hash = os.environ.get("TG_HASH")
    phone_number = os.environ.get("PHONE")
    password = os.environ.get("PASSWORD")
    
    parser = TelegramParser(int(api_id), api_hash, phone_number, password)
    with parser.client:
        parser.client.loop.run_until_complete(
            telegram_parse(parser,
                           count_items=1000,
                           search='МРИЯ',
                           min_date=datetime(year=2024, month=10, day=12),
                           filename='mriya_messages.csv'))


if __name__ == '__main__':
    # with client:
    #     client.loop.run_until_complete(main())
    # main()
    """
    МРИЯ / 6
    Отель Мрия / 1
    Мрия курорт / 4
    """
