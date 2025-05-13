from src.get_info.parsers.yandex_maps import YandexMapsParser, YandexMapsConfig
from src.get_info.core import MasterParserConfig

if __name__ == "__main__":
    global_config = MasterParserConfig(sort_type="Сначала положительные")
    local_config = YandexMapsConfig(1303073708)
    parser = YandexMapsParser(local_config)
    result = parser.parse(global_config)
    print(result)
