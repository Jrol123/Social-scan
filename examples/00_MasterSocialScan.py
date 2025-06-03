from dotenv import dotenv_values
import os

ddl_path = r"D:\Hardcode_libs\GTK3-Runtime Win64\bin"

os.add_dll_directory(ddl_path)

from src.core import MasterScan, MasterScanConfig
from src.get_info.parsers.yandex_maps import YandexMapsParser, YandexMapsConfig
from src.get_info.core import MasterParserConfig, MasterParser


if __name__ == "__main__":
    secrets = dotenv_values()

    cache_dir = "D:/TRANSFORMERS_MODELS"

    global_config = MasterParserConfig(sort_type="rating_ascending", count_items=10)
    local_config = YandexMapsConfig(1303073708)
    yp = YandexMapsParser(local_config)
    parser = MasterParser(yp)

    metadata = {
        "company": "МРИЯ",
        "description": "курорт премиум-класса на южном берегу Крыма",
    }

    scanConfig = MasterScanConfig(parser, metadata, cache_dir, global_config)

    scan = MasterScan(scanConfig)

    scan.generate_report(
        "report.pdf", secrets["MISTRAL_API_TOKEN"], secrets["CHUTES_API_TOKEN"]
    )
