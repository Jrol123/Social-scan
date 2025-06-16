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
    
    cache_dir = None  # "D:/TRANSFORMERS_MODELS"
    
    global_config = MasterParserConfig(
        sort_type="date_descending",
        min_date=datetime(2025, 1, 1),
        count_items=1000,
    )
    
    local_gm_config = GoogleMapsConfig(
        r"https://www.google.com/maps/place/МАНЖЕРОК,+Курорт/@51.8155393,85.807883,17.06z/data=!4m11!3m10!1s0x42c45b01fc3f5529:0xcb4eee4c87c48bb6!5m2!4m1!1i2!8m2!3d51.8154038!4d85.8089071!9m1!1b1!16s%2Fg%2F11h1hyp7_?hl=ru&entry=ttu&g_ep=EgoyMDI1MDYxMS4wIKXMDSoASAFQAw%3D%3D"
    )
    local_ov_config = OtzovikConfig(
        "https://otzovik.com/reviews/gornolizhniy_kompleks_manzherok_russia_gorniy_altay_manzherok/?ratio=N&order=date_desc"
    )
    # local_vk_config = VKConfig(q="МРИЯ -купить")
    local_tg_config = TelegramConfig(q="Манжерок", channel_list=["t.me/gkmanjerok"])
    local_ym_config = YandexMapsConfig(1038646745)
    
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
        "company": "Манжерок",
        "description": "Горнолыжный комплекс, место для семейного отдыха.",
        "idx_to_service": {service.service_id: str(service)
                           for service in parser.parsers}
    }
    
    scanConfig = MasterScanConfig(
        parser, metadata, cache_dir, global_config,
        MasterSentimentConfig(
            modelPath="sismetanin/rubert-ru-sentiment-rusentiment"))
    
    scan = MasterScan(scanConfig)
    
    scan.generate_report(
        "report4.pdf", secrets["MISTRAL_API_TOKEN"], secrets["CHUTES_API_TOKEN"]
    )
