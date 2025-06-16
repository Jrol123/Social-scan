from datetime import datetime
from dotenv import dotenv_values
# import os

# ddl_path = r"D:\Hardcode_libs\GTK3-Runtime Win64\bin"
#
# os.add_dll_directory(ddl_path)

from src.core import MasterScan, MasterScanConfig
from src.get_info.parsers.google_maps import GoogleMapsParser, GoogleMapsConfig
from src.get_info.parsers.otzovik import OtzovikParser, OtzovikConfig
from src.get_info.parsers.vk import VKParser, VKConfig
from src.get_info.parsers.telegram import TelegramParser, TelegramConfig
from src.get_info.parsers.yandex_maps import YandexMapsParser, YandexMapsConfig
from src.get_info.core import MasterParserConfig, MasterParser
from src.get_labels.transformers.sentiment import MasterSentimentConfig

if __name__ == "__main__":
    secrets = dotenv_values()

    cache_dir = None # "D:/TRANSFORMERS_MODELS"

    global_config = MasterParserConfig(
        sort_type="date_descending", min_date= datetime(2024, 1, 1),
        max_date=datetime(2025, 5, 18), count_items=2000,
    )  # sort_type="rating_ascending", count_items=10)
    
    local_gm_config = GoogleMapsConfig(
        r"https://www.google.com/maps/place/?q=place_id:ChIJ7WjSWynClEARUUiva4PiDzI"
    )
    local_ov_config = OtzovikConfig(
        "https://otzovik.com/reviews/sanatoriy_mriya_resort_spa_russia_yalta/"
    )
    # local_vk_config = VKConfig(q="МРИЯ -купить")

    local_tg_config = TelegramConfig("МРИЯ", ["t.me/mriyaresortchat"])
    local_ym_config = YandexMapsConfig(1303073708)
    
    gmp = GoogleMapsParser(local_gm_config)
    ovp = OtzovikParser(local_ov_config)
    # vkp = VKParser(secrets["VK_TOKEN"], local_vk_config)
    tgp = TelegramParser(
        local_tg_config,
        int(secrets["TG_ID"]),
        secrets["TG_HASH"],
        secrets["PHONE"],
        secrets.get("PASSWORD"),
    )
    ymp = YandexMapsParser(local_ym_config)
    
    parser = MasterParser(gmp, ovp, tgp, ymp)

    metadata = {
        "company": "МРИЯ",
        "description": "курорт премиум-класса на южном берегу Крыма",
        "idx_to_service": {service.service_id: str(service)
                           for service in parser.parsers}
    }

    scanConfig = MasterScanConfig(
        parser, metadata, cache_dir, global_config,
        MasterSentimentConfig(
            modelPath="sismetanin/rubert-ru-sentiment-rusentiment"))

    scan = MasterScan(scanConfig)

    scan.generate_report(
        "report3.pdf", secrets["MISTRAL_API_TOKEN"], secrets["CHUTES_API_TOKEN"]
    )
