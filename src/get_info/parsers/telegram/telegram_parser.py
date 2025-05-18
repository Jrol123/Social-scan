import os
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient

from ...abstract import AsyncParser
from ...core import MasterParserConfig
from .config import TelegramConfig


class TelegramParser(AsyncParser):
    def __init__(
        self,
        local_config: TelegramConfig,
        api_id: int,
        api_hash,
        phone,
        password=None,
        session_name="monitoring",
        system_version="Windows 10", # 4.16.30-vxCUSTOM
        app_version="4.16.3",
        device_model="PC"
    ):
        super().__init__(4, local_config)

        self.client = TelegramClient(
            session_name,
            api_id,
            api_hash,
            system_version=system_version,
            app_version=app_version,
            device_model=device_model
        )
        self.phone = phone
        self.password = password
        # self.client.start(phone=phone, password=password)
        # assert self.client.connect()
        # if not self.client.is_user_authorized():
        print("ssSSss")

    async def parse(
        self, global_conifg: MasterParserConfig
    ) -> list[dict[str, str | int | float | None]]:
        print("ssSSss")
        # assert await self.client.connect()
        # if not self.client.is_user_authorized():
        #     await self.client.send_code_request(self.phone)
        #     await self.client.sign_in(self.phone, input("Enter code: "))
            
        await self.client.start(phone=self.phone, password=self.password)
        q = self.config.q
        channels_list = self.config.channels_list
        wait_sec = self.config.wait_sec
        count_items = global_conifg.count_items
        if isinstance(channels_list, str):
            channels_list = open(self.config.channels_list).readlines()
            channels_list = list(map(str.strip, channels_list))
        else:
            channels_list = self.config.channels_list

        min_date = self._date_convert(global_conifg.min_date, datetime)
        max_date = self._date_convert(global_conifg.max_date, datetime)

        data = []
        for channel in channels_list:
            data.extend(
                await self.get_channel_history(
                    channel, q, count_items, min_date, max_date, wait_sec
                )
            )
            time.sleep(wait_sec)

        if not data:
            print("There is no data collected from telegram.")
            return []

        return data

    async def get_channel_history(
        self,
        channel_link,
        q,
        count_items: int,
        min_date: datetime,
        max_date: datetime,
        wait_sec: int,
    ):
        data = []
        try:
            channel = await self.client.get_entity(channel_link)
        except ValueError:
            return []
        
        num_messages = 200
        channel_id = channel.id
        print(channel_link)
        if count_items == -1:
            channel_history = [
                message
                async for message in self.client.iter_messages(
                    channel_link, limit=num_messages, search=q, offset_date=max_date
                )
            ]
            while channel_history:
                time.sleep(wait_sec)
                for message in channel_history:
                    if message.text:
                        data.append(await self.__form_line(message, channel_id))

                last_date = data[-1]["date"]
                channel_history = [
                    message
                    async for message in self.client.iter_messages(
                        channel_link, limit=num_messages,
                        offset_date=last_date, search=q
                    )
                ]
                if min_date is not None:
                    channel_history = list(
                        filter(
                            lambda m: m.date.replace(tzinfo=None) > min_date,
                            channel_history,
                        )
                    )
        else:
            async for message in self.client.iter_messages(
                channel_link, limit=num_messages, search=q, offset_date=max_date
            ):
                if message.text:
                    data.append(await self.__form_line(message, channel_id))

                if len(data) >= count_items:
                    break

            offset_data = None
            i = 0
            while len(data) < count_items:
                time.sleep(1)
                i += num_messages
                try:
                    offset_data = data[-1]["date"]
                except IndexError:
                    pass

                async for message in self.client.iter_messages(
                    channel_link,
                    limit=num_messages if i + num_messages < count_items
                          else (i - count_items) % num_messages,
                    offset_date=offset_data,
                    search=q,
                ):
                    if message.text:
                        data.append(await self.__form_line(message, channel_id))

                    if len(data) >= count_items:
                        break

                if min_date is not None and data and data[-1]["date"] < min_date:
                    break
        
        for i in range(len(data)):
            data[i]["date"] = int(data[i]["date"].timestamp())
        
        return data

    async def __form_line(self, message, channel_id):
        try:
            # TODO: оптимизировать получение объекта пользователя путём кэширования
            user = await self.client.get_entity(message.from_id)
            user_id = user.id
        except (TypeError, ValueError, AttributeError):
            user_id = None

        return {
            "service_id": self.service_id,
            "name": user_id or channel_id,
            "additional_id": channel_id if user_id else None,
            "date": message.date.replace(tzinfo=None),
            "rating": None,
            "text": message.text,
            "answer": None,
        }


def save_reviews_to_csv(reviews, filename="telegram_reviews.csv"):
    if not reviews:
        return

    df = pd.DataFrame(reviews)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    df.to_csv(filename, index=False, encoding="utf-8")


async def telegram_parse(
    parser: TelegramParser,
    channels_list="channel_list.txt",
    search=None,
    count_items=100,
    min_date=None,
    max_date=None,
    filename="telegram_reviews.csv",
):
    params = {"channels_list": channels_list, "q": search, "count_items": count_items}
    if min_date:
        params["min_date"] = min_date
    if max_date:
        params["max_date"] = max_date

    messages = await parser.parse(**params)
    save_reviews_to_csv(messages, filename)


def main():
    load_dotenv()

    api_id = os.environ.get("TG_ID")
    api_hash = os.environ.get("TG_HASH")
    phone_number = os.environ.get("PHONE")
    password = os.environ.get("PASSWORD")

    parser = TelegramParser(api_id, api_hash, phone_number, password)
    # with parser.client:
    parser.client.loop.run_until_complete(
        telegram_parse(
            parser,
            count_items=1000,
            search="МРИЯ",
            min_date=datetime(year=2024, month=10, day=12),
            filename="mriya_messages.csv",
        )
    )


if __name__ == "__main__":
    # with client:
    #     client.loop.run_until_complete(main())
    # main()
    """
    МРИЯ / 6
    Отель Мрия / 1
    Мрия курорт / 4
    """
