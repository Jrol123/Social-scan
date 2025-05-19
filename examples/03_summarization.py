"""
Этот пример посвящён суммаризации, третьему этапу в pipeline.

Здесь происходит суммаризация отзывов.
"""
import os
from dotenv import load_dotenv

load_dotenv()
chutes_token = os.environ.get("CHUTES_API_TOKEN")
if not chutes_token:
    raise ValueError(
        "CHUTES_API_TOKEN is missing. Please set it in the environment variables."
    )

mistral_token = os.environ["MISTRAL_API_TOKEN"]
if not mistral_token:
    raise ValueError(
        "MISTRAL_API_TOKEN is missing. Please set it in the environment variables."
    )
