import vk_api
from dotenv import dotenv_values

secrets = dotenv_values(".env")
"""Секреты"""

vk = vk_api.VkApi(token=secrets["VK_TOKEN"])
"""Модуль ВК"""


def search_feed(
    q: str,
    count: int,
    list: int = 0,
    geo: tuple[int, int] | None = None,
    time: tuple[int | None, int | None] | None = None,
):
    params = {"q": q, "count": count, "start_from": list}
    # ! Времянка
    if geo:
        params["geo"] = geo
    if time:
        params["time"] = time
    return vk.method("newsfeed.search", values=params)


# Обратите внимание — даже при использовании параметра offset для получения информации доступны только первые 1000 результатов.
# start_from - offset по страницам.
# Поэкспериментировать с offset и start_from, потому как они, похоже, не эквивалентны
# Или просто реальное количество записей не совпадает с видимым
# Или ограничение на 1000 записей очень жёсткое
# print(len(search_feed("кино", 1)['items']))
# print(search_feed("кино", 1, 0)['items'][0]['id'], search_feed("кино", 1, 1)['items'][0]['id'])
zp = search_feed("кино", 1, 199)
print(len(zp["items"]))
for i in zp["items"]:
    print(i["id"])

# total_count не соответствует настоящему count-у
