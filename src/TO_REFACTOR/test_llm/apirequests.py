import asyncio
import os
import time

import pandas as pd
from dotenv import load_dotenv

from ...apirequests import invoke_chute, invoke_mistral

load_dotenv()

chutes_token = os.environ.get("CHUTES_API_TOKEN")
mistal_token = os.environ.get("MISTRAL_API_TOKEN")

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

# ---- ВСПОМАГАТЕЛНЫЕ ФУНКЦИИ ДЛЯ РЕШЕНИЯ ЗАДАЧ


def multilabel_classification(
    reviews: pd.DataFrame,
    categories: list,
    api_token: str,
    metadata: dict,
    model_name="deepseek-ai/DeepSeek-V3-0324",
    batch_size=24000
) -> list[dict[str, str | None]]:
    df = reviews[['text']].reset_index(drop=True)
    df = df.dropna(how="all")
    df["len"] = df["text"].str.len()
    df["cumlen"] = df["len"].cumsum()
    df["cumlen"] = df["cumlen"] + [6 * i for i in range(len(df))]
    
    instr = (
        "Ты - опытный помощник по выявлению проблем компании "
        f"\"{metadata['company']}\", на которые жалуются клиенты в своих отзывах. "
        f"Вот краткое описание компании: {metadata['description']}.\n\n"
        "Твоя задача - максимально точно перечислить "
        "все конкретные проблемы и жалобы, упоминаемые пользователем, связанные "
        "с бизнесом, не теряя уточняющие детали. "
        "Каждую упоминаемую проблему отнеси к одному из предложенных классов: "
        f"{', '.join(categories)}, остальные - если нет проблем, "
        'которые относятся к классу, ставь символ "-", '
        "и если проблема не относится ни к одному классу, "
        'относи её к классу "остальные".'
        "Соблюдай шаблон ввода:\n1. текст отзыва\n----\n"
        "2. текст отзыва\n----\n... (все оставшиеся отзывы)\n----\n"
        "n. текст отзыва\n\nШаблон вывода:\n\n1.\n"
    )
    
    instr += "\n".join(
        [k + f': проблема1; проблема2, связанных с "{k}" в отзыве 1\n'
         for k in categories]
    )
    instr += (
        "\nостальные: перечисление проблем в первом отзыве, не относящихся "
        "ни к одному из классов выше\n\n"
        "... (все остальные отзывы)\n\nn.\n"
    )
    instr += "\n".join(
        [k + f': перечисление проблем, связанных с "{k}" ' f"в отзыве n\n"
         for k in categories]
    )
    instr += (
        "\nостальные: перечисление проблем в последнем отзыве, не относящихся "
        "ни к одному из классов выше"
    )
    
    i = 0
    outputs = []
    while i * batch_size <= df.iloc[-1, -1]:
        batch = df.loc[(i * batch_size < df['cumlen'])
                       & (df['cumlen'] < (i + 1) * batch_size), "text"]
        batch = [str(j + 1) + '. ' for j in range(len(batch))] + batch
        prompt = "\n----\n".join(batch)
        # print(instr)
        # print(prompt)
        
        time.sleep(10)
        try:
            if 'mistral' not in model_name:
                output = asyncio.run(
                    invoke_chute(prompt, instr, api_token, model_name)
                )
                if '</think>' in output:
                    output = output.split('</think>', 1)[1]
            else:
                output = asyncio.run(
                    invoke_mistral(prompt, instr, api_token, model_name)
                )
            
        except asyncio.exceptions.TimeoutError:
            continue
        
        if not output:
            continue
        
        outputs.append(output)
        i += 1
    
    outputs = [s.strip().split('.\n', 1)[1] if '.\n' in s
               else s.strip().split('. ', 1)[1]
               for part in outputs for s in part.split('\n\n')]
    
    problems = []
    for review in outputs:
        review = review
        review_categories = review.split('\n') if '\n' in review else [review]
        review_cats = dict.fromkeys(categories + ['остальные'], None)
        for category in review_categories:
            name, enum_problems = category.split(': ', 1)
            if name in review_cats:
                review_cats[name] = (enum_problems if enum_problems.strip() != '-'
                                     else None)

        problems.append(review_cats)
        
    return problems


if __name__ == "__main__":
    ...
