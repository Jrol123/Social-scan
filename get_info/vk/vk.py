import vk_api
from datetime import datetime
from time import sleep
from ..abstract import Parser

SERVICE_INDEX = 5
MAX_COUNT = 200


class VKParser(Parser):

    def __init__(self, vk_token: str | tuple[str, str]):
        super().__init__(SERVICE_INDEX)
        if isinstance(vk_token, str):
            self.vk: vk_api.VkApi = vk_api.VkApi(token=vk_token)
        elif isinstance(vk_token, tuple):
            self.vk = vk_api.VkApi(*vk_token)
            # Вход по login - password. Лучше всего использовать номер телефона как логин
        else:
            raise TypeError("vk_token must be a str or a tuple of (login, password)")

    def parse(
        self,
        q: str,
        count_items: int = 1000,
        start_from: str = "0",
        min_date: int = int(datetime(1970, 1, 16).timestamp()),
        max_date: int = int(datetime.now().timestamp()),
        fields: str = "id, first_name, last_name",
        return_count: bool = False,
    ) -> list[dict[str, list[dict[str, str | int]]]] | int:
        min_date = self._date_convert(min_date, int)
        max_date = self._date_convert(max_date, int)
        if count_items == -1:
            res = self.vk.method(
                "newsfeed.search",
                values={
                    "q": q,
                    "count": 1,
                    "start_time": min_date,
                    "end_time": max_date,
                },
                raw=True,
            )
            count_items = res["response"]["total_count"]
            if return_count:
                return count_items

        # TODO: Переделать print под logging
        print(f"total_count: {count_items}")

        result = {"items": [], "profiles": [], "groups": []}
        while count_items != 0:
            #! Как определять, когда записей действительно нет, а когда это просто ошибка/временное ограничение?
            count_items, cur_result = self.__search(
                q, count_items, start_from, min_date, max_date, fields
            )
            result = self.__combine_result(result, cur_result)
            print(f"rem_count: {count_items}\tlen: {len(cur_result['items'])}")
            if len(cur_result["items"]) == 0:
                print("no data!\nretrying...")
                sleep(60)  # TODO: Поэкспериментировать с задержкой
            else:
                if len(cur_result["items"]) == 1:
                    # print(cur_result)
                    pass
                last_date = cur_result["items"][-1]["date"]
                max_date = last_date

        print(count_items)
        
        return result["items"]

    def __search(
        self,
        q: str,
        total_count: int = 1000,
        start_from: str = "0",
        start_time: int | None = 0,
        end_time: int = int(datetime.now().timestamp()),
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
        #! Иногда просто перестаёт высылать результаты. Антибот?
        result = self.vk.method("newsfeed.search", values=params)
        total_count -= len(result["items"])
        if total_count != 0 and "next_from" in result.keys():
            rem_count, next_result = self.__search(
                q, total_count, result["next_from"], start_time, end_time, fields
            )
            # self.__clean_result(next_result)
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
    """
    owner_id - сообщество, from_id - тот, кто публиковал от сообщества
    
    from_id - name
    owner_id - additional_id
    """
    # TODO: Чистить
    def __clean_result(self, result: dict[str, list[dict[str, str | int]]]) -> None:
        def __key_clean(d: dict[str, str | list], save_keys: list[str]) -> None:
            # Получаем список ключей словаря
            keys = list(d.keys())
            # Удаляем все ключи, кроме нужных
            for key in keys:
                if key not in save_keys:
                    del d[key]

        def __clean(toClean_dict: list[dict[str, str | list]], save_keys: list[str]) -> None:
            for d in toClean_dict:
                __key_clean(d, save_keys)

        rest_keys_result = ["items", "profiles", "groups"]
        rest_keys_items = ["date", "edited", "from_id", "owner_id", "text"]
        rest_keys_profiles = ["id", "first_name", "last_name"]
        rest_keys_groups = ["id", "name"]

        items: list[dict] = result["items"]
        profiles: list[dict] = result["profiles"]
        groups: list[dict] = result["groups"]

        __key_clean(result, rest_keys_result)
        __clean(items, rest_keys_items)
        __clean(profiles, rest_keys_profiles)
        __clean(groups, rest_keys_groups)

        item_rename_keys = {"owner_id": "additional_id", "from_id": "name"}

        for item in items:
            item["from_id"] = str(item["from_id"])
            item["owner_id"] = str(item["owner_id"])

            # Если группа - оставляем owner_id, чтобы если что, то делать фильтр по группам.
            if item["from_id"] == item["owner_id"] and item["owner_id"][0] != "-":
                item["owner_id"] = None

            item["service_id"] = self.service_id
            item["rating"] = None
            item["answer"] = None

            for key, rename_key in item_rename_keys.items():
                item[rename_key] = item[key]
            for key in item_rename_keys.keys():
                del(item[key])


"""
"service_id" (int): Внутренний индекс сервиса
"name" (str): Имя пользователя. Для Telegram и VK хранить id пользователя.
"additional_id" (str | None): Дополнительный идентификатор для уточнения сообщения (пример: канал в Telegram).
"date" (int): Дата в формате timestamp.
"rating" (float | None): Рейтинг (1.0-5.0, если есть, иначе None).
"text" (str): Текст отзыва.
"answer" (str | None): Ответ на отзыв (если присутствует).
"""
