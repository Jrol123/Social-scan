"""
Этот пример посвящён суммаризации, третьему этапу в pipeline.

Здесь происходит суммаризация отзывов.
"""

from pandas import read_csv
from dotenv import dotenv_values
from src.get_summarization import summarize_reviews

DEFAULT_INSTRUCTION = (
    "Ты - опытный помощник по выявлению проблем бизнеса, "
    "на которые жалуются клиенты в своих отзывах. "
    "Твоя задача - максимально точно перечислить все конкретные "
    "проблемы и жалобы, упоминаемые пользователем, связанные "
    "с бизнесом, не теряя уточняющие детали. "
    "Сокращай объём текста минимум до 256 символов. "
    "Соблюдай шаблон ввода:\n\n1. текст отзыва"
    "\n----\n2. текст отзыва\n----\n"
    "... (все оставшиеся отзывы)\n----\nn. текст отзыва\n\n"
    "Шаблон вывода:\n\n1. суммаризация первого отзыва\n"
    "----\n2. суммаризация второго отзыва\n----\n"
    "... (суммаризация всех оставшихся отзывов)\n----"
    "\nn. суммаризация последнего отзыва"
)

if __name__ == "__main__":
    secrets = dotenv_values()

    reviews = read_csv("examples/02a_example_filtered_data.csv", index_col=0)
    summaries = summarize_reviews(
        reviews,
        model_name="mistral",
        token=secrets["MISTRAL_API_TOKEN"],
        instr=DEFAULT_INSTRUCTION,
    )
    reviews["summary"] = summaries
    reviews[["text", "summary"]].to_csv("examples/03_summarized_data.csv")
