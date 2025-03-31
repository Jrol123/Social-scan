import vk_api
import time
from dotenv import dotenv_values

secrets = dotenv_values(".env")
"""Секреты"""

# vk = vk_api.VkApi(token=secrets["VK_TOKEN"])
# """Модуль ВК"""


class VK_parser:
    def __init__(self, vk_token: str | tuple[str, str]):
        if isinstance(vk_token, str):
            self.vk = vk_api.VkApi(token=vk_token)
        elif isinstance(vk_token, tuple[str, str]):
            self.vk = vk_api.VkApi(*vk_token)
        else:
            raise TypeError

    def search_feed(
        self,
        q: str,
        count: int = 200,
        start_from: int = 0,
        time_start: int | None = 0,
        time_end: int = int(time.time()),
    ):
        params = {
            "q": q,
            "count": count,
            "start_from": start_from,
            "time_start": time_start,
            "time_end": time_end,
            "extended": True,
        }
        return self.vk.method("newsfeed.search", values=params)

# print(len(search_feed("кино", 1)['items']))
# print(search_feed("кино", 1, 0)['items'][0]['id'], search_feed("кино", 1, 1)['items'][0]['id'])

vk = VK_parser(secrets["VK_TOKEN"])
zp = vk.search_feed("кино")
print(len(zp["items"]))
for i in zp["items"]:
    print(i["id"])
    
386598
62063

4958021
4958002