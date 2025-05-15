from src.get_info.parsers.yandex_maps import YandexMapsParser, YandexMapsConfig
from src.get_info.core import MasterConfig

# import sys
# import os

# # Добавляем корень проекта в путь поиска модулей
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

if __name__ == "__main__":
    # yandex
    global_config = MasterConfig(sort_type="Сначала положительные")
    local_config = YandexMapsConfig(1303073708)
    parser = YandexMapsParser(local_config)
    result = parser.parse(global_config)
    print(result)
