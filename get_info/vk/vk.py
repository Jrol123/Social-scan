import vk_api
import time
from dotenv import dotenv_values

MAX_COUNT = 200


class VK_parser:

    def __init__(self, vk_token: str | tuple[str, str]):
        if isinstance(vk_token, str):
            self.vk = vk_api.VkApi(token=vk_token)
        elif isinstance(vk_token, tuple):
            self.vk = vk_api.VkApi(*vk_token)
        else:
            raise TypeError

    def search_feed(
        self,
        q: str,
        total_count: int = 1000,
        start_from: str = "0",
        start_time: int | None = 0,
        end_time: int = int(time.time()),
        fields: str = "id, first_name, last_name",
    ) -> dict[str, list[dict[str, str | int]]]:
        if total_count == -1:
            total_count = self.vk.method(
                "newsfeed.search",
                values={
                    "q": q,
                    "count": 1,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )["total_count"]
        print(f"total_count: {total_count}")
        result = {"items": [], "profiles": [], "groups": []}
        while total_count != 0:
            total_count, cur_result = self.__search(
                q, total_count, start_from, start_time, end_time, fields
            )
            result = self.__combine_result(result, cur_result)
            print(f"rem_count: {total_count}\tlen: {len(cur_result['items'])}")
            try:
                last_date = cur_result["items"][-1]["date"]
                end_time = last_date
            except:
                print("no data!")
        print(total_count)
        return result

    def __search(
        self,
        q: str,
        total_count: int = 1000,
        start_from: str = "0",
        start_time: int | None = 0,
        end_time: int = int(time.time()),
        fields: str = "id, first_name, last_name",
    ) -> tuple[int, dict[str, list[dict[str, str | int]]]]:
        get_param = min(MAX_COUNT, total_count)
        if get_param == 0:
            print("АЛЯРМ! next_from соглал!")
            return total_count, {"items": [], "profiles": [], "groups": []}

        params = {
            "q": q,
            "count": get_param,
            "start_from": start_from,
            "start_time": start_time,
            "end_time": end_time,
            "extended": True,
            "fields": fields,
        }
        result = self.vk.method("newsfeed.search", values=params)
        total_count -= len(result["items"])
        #! Почему-то получаются те же результаты
        if total_count != 0 and "next_from" in result.keys():
            rem_count, next_result = self.__search(
                q, total_count, result["next_from"], start_time, end_time
            )
            self.__clean_result(next_result)
            self.__clean_result(result)

            finish_res = self.__combine_result(result, next_result)

            return rem_count, finish_res
        self.__clean_result(result)
        return total_count, result

    def __combine_result(
        self, res1: dict[str, list[dict]], res2: dict[str, list[dict]]
    ) -> dict[str, list[dict]]:
        def __combinator(list1: list[dict], list2: list[dict]) -> list[dict]:
            seen = set()
            combined = []
            for d in list1 + list2:
                # Сортируем элементы словаря для однозначности
                dict_tuple = tuple(sorted(d.items()))
                if dict_tuple not in seen:
                    seen.add(dict_tuple)
                    combined.append(d)
            return combined

        finish_res = {}
        for name in ["profiles", "groups"]:
            finish_res[name] = __combinator(res1[name], res2[name])
        finish_res["items"] = res1["items"] + res2["items"]
        return finish_res

    def __clean_result(self, result: dict[str, list[dict[str, str]]]) -> None:
        def __key_clean(d: dict[str, str | list], save_keys: list[str]) -> None:
            # Получаем список ключей словаря
            keys = list(d.keys())
            # Удаляем все ключи, кроме нужных
            for key in keys:
                if key not in save_keys:
                    del d[key]

        def __clean(toClean_dict: list[dict[str, str]], save_keys: list[str]) -> None:
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

secrets = dotenv_values(".env")
"""Секреты"""
vk = VK_parser(secrets["VK_TOKEN"])
zp = vk.search_feed(total_count=10**4, q="кино")
print(len(zp["items"])) # 4960210
# for i in zp["items"]:
#     print(i["id"])
