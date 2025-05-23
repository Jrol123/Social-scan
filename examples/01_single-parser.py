"""
Этот пример посвящён парсингу, первому этапу в pipeline.

В данном примере показано, как работать с парсерами по-отдельности.
"""
from src.get_info.parsers.yandex_maps import YandexMapsParser, YandexMapsConfig
from src.get_info.core import MasterParserConfig


if __name__ == "__main__":
    # yandex
    global_config = MasterParserConfig(sort_type="rating_ascending")
    local_config = YandexMapsConfig(1303073708)
    parser = YandexMapsParser(local_config)
    result = parser.parse(global_config)
    print(result)
