import vk_api
import time
from dotenv import dotenv_values

MAX_COUNT = 200

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
        total_count: int = 1000,
        start_from: str = '0',
        start_time: int | None = 0,
        end_time: int = int(time.time()),
        fields: str = "id, first_name, last_name",
    ) -> dict[str:dict]:
        get_param = min(MAX_COUNT, total_count)
        if get_param == 0:
            print("АЛЯРМ! next_from соглал!")
            return

        params = {
            "q": q,
            "count": get_param,
            "start_from": start_from,
            "start_time": start_time,
            "end_time": end_time,
            "extended": True,
            "fields": fields,
        }
        total_count -= get_param
        result: dict[str:dict] = self.vk.method("newsfeed.search", values=params)
        #! Почему-то получаются те же результаты
        if total_count != 0 and "next_from" in result.keys():
            next_result = self.search_feed(q, total_count, result["next_from"], start_time, end_time)
            self.__clean_result(next_result)
            self.__clean_result(result)
            finish_res = {}
            for name in result.keys():
                finish_res[name] = self.__combinator(result[name], next_result[name])
            return finish_res
        self.__clean_result(result)
        return result
        
    def __combinator(self, list1: list[dict], list2: list[dict]) -> list[dict]:
        seen = set()
        combined = []
        for d in list1 + list2:
            # Сортируем элементы словаря для однозначности
            dict_tuple = tuple(sorted(d.items()))
            if dict_tuple not in seen:
                seen.add(dict_tuple)
                combined.append(d)

        return combined

    def __clean_result(self, result: dict[str:dict]) -> None:
        def __key_clean(d: dict, save_keys: list[str]) -> None:
            # Получаем список ключей словаря
                keys = list(d.keys())
                # Удаляем все ключи, кроме нужных
                for key in keys:
                    if key not in save_keys:
                        del d[key]
        def __clean(toClean_dict: dict, save_keys: list[str]) -> None:
            for d in toClean_dict:
                __key_clean(d, save_keys)
                

        rest_keys_result = ["items", "profiles", "groups"]
        rest_keys_items = ["id", "date", "edited", "owner_id", "text"]
        rest_keys_profiles = ["id", "first_name", "last_name"]
        rest_keys_groups = ["id", "name"]
        

        items: list[dict] = result["items"]
        profiles: list[dict] = result["profiles"]
        groups: list[dict] = result["groups"]
        
        __key_clean(result, rest_keys_result)
        __clean(items, rest_keys_items)
        __clean(profiles, rest_keys_profiles)
        __clean(groups, rest_keys_groups)


# print(len(search_feed("кино", 1)['items']))
# print(search_feed("кино", 1, 0)['items'][0]['id'], search_feed("кино", 1, 1)['items'][0]['id'])

vk = VK_parser(secrets["VK_TOKEN"])
zp = vk.search_feed("кино")
print(len(zp["items"]))
# for i in zp["items"]:
#     print(i["id"])
