import vk_api
from dotenv import dotenv_values

secrets = dotenv_values(".env")
"""Секреты"""

vk = vk_api.VkApi(
    token=secrets['VK_TOKEN'])
"""Модуль ВК"""

def search_feed(q: str, count: int, list, geo: tuple[int, int], time: tuple[int|None, int|None])
params = {}
vk.method('newsfeed.search', values=params)
# Обратите внимание — даже при использовании параметра offset для получения информации доступны только первые 1000 результатов.
# start_from - offset по страницам.
# Поэкспериментировать с offset и start_from, потому как они, похоже, не эквивалентны
# Или просто реальное количество записей не совпадает с видимым
# Или ограничение на 1000 записей очень жёсткое