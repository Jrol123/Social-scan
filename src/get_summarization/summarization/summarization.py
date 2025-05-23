import asyncio
import pandas as pd

from ...apirequests import invoke_chute, invoke_mistral

NNAME = {
    "mistral": [invoke_mistral, "mistral-small-latest"],
    "chute": [invoke_chute, "deepseek-ai/DeepSeek-V3-0324"],
    "deepseek": [invoke_chute, "deepseek-ai/DeepSeek-V3-0324"],
}

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


def summarize_reviews(
    reviews: pd.DataFrame,
    model_name: str,
    token,
    instr = DEFAULT_INSTRUCTION,
    batch_size: int = 32000,
):
    df = reviews[["text"]].reset_index(drop=True)
    df = df.dropna(how="all")
    df["len"] = df["text"].str.len()
    df["cumlen"] = df["len"].cumsum()
    df["cumlen"] = df["cumlen"] + [6 * i for i in range(len(df))]

    i = 0
    outputs = []
    while i * batch_size <= df.iloc[-1, -1]:
        batch = df.loc[
            (i * batch_size < df["cumlen"]) & (df["cumlen"] < (i + 1) * batch_size),
            "text",
        ]
        batch = [str(j + 1) + ". " for j in range(len(batch))] + batch
        prompt = "\n----\n".join(batch)

        if model_name in NNAME.keys():
            output = asyncio.run(
                NNAME[model_name][0](
                    query=prompt,
                    instruction=instr,
                    token=token,
                    model=NNAME[model_name][1],
                )
            )
        else:
            raise ValueError("Неправильное имя модели!")
        if output:
            outputs.append(output)
        else:
            continue

        i += 1

    output = "\n----\n".join(outputs)
    output = output.split("\n----\n")
    if "1. " in output[0]:
        output = [summary.split(". ", 1)[1] for summary in output]
    elif "1.\n" in output[0]:
        output = [summary.split(".\n", 1)[1] for summary in output]

    return output


def gen_categories(
    reviews: pd.DataFrame,
    model_name: str,
    token,
    metadata: dict,
    batch_size: int = 32000,
):
    instr1 = (f"Ты - аналитик отзывов о компании \"{metadata['company']}\"."
              f"Вот краткое описание компании: {metadata['description']}.\n\n"
              "Проанализируй следующие пользовательские отзывы и выдели основные "
              "категории проблем, которые в них упоминаются, учитывая специфику "
              "области компании.\n\nСоблюдай шаблон ввода:\n\nтекст отзыва 1"
              "\n----\nтекст отзыва 2\n----\n... (все оставшиеся отзывы)\n----\n"
              "текст отзыва n\n\nШаблон вывода:\n\n1. Категория 1\n2. Категория 2"
              "\n...\nk. Категория k"
             )
    
    df = reviews[["text"]].reset_index(drop=True)
    df = df.dropna(how="all")
    df["len"] = df["text"].str.len()
    df["cumlen"] = df["len"].cumsum()
    df["cumlen"] = df["cumlen"] + [6 * i for i in range(len(df))]

    i = 0
    outputs = []
    while i * batch_size <= df.iloc[-1, -1]:
        batch = df.loc[
            (i * batch_size < df["cumlen"]) & (df["cumlen"] < (i + 1) * batch_size),
            "text",
        ]
        batch = [str(j + 1) + ". " for j in batch.index] + batch
        prompt = "\n----\n".join(batch)

        if model_name in NNAME.keys():
            output = asyncio.run(
                NNAME[model_name][0](
                    query=prompt,
                    instruction=instr1,
                    token=token,
                    model=NNAME[model_name][1],
                )
            )
        else:
            raise ValueError("Неправильное имя модели!")
        
        if output:
            outputs.append(output)
        else:
            continue

        i += 1

    output = "\n\n".join(outputs)
    output = output.split("\n\n")
    output = [category.split(". ", 1)[1].split('\n', 1)[0]
              .replace('**', '').replace(':', '')
              for category in output if category[0].isdigit()]
    
    instr2 = (f"Ты - аналитик отзывов о компании \"{metadata['company']}\"."
              f"Вот краткое описание компании: {metadata['description']}.\n\n"
              "Выдели непересекающиеся категории, сформированные по отзывам.\n\n"
              "Соблюдай шаблон ввода:\n\n1. Категория 1\n2. Категория 2"
              "\n...\nk. Категория k"
              "Шаблон вывода:\n\n1. Категория 1\n2. Категория 2"
              "\n...\nk. Категория k-n")
    
    prompt = "\n".join([str(i + 1) + '. ' + cat for i, cat in enumerate(output)])
    while True:
        if model_name in NNAME.keys():
            output = asyncio.run(
                NNAME[model_name][0](
                    query=prompt,
                    instruction=instr2,
                    token=token,
                    model=NNAME[model_name][1],
                )
            )
        else:
            raise ValueError("Неправильное имя модели!")
        
        if not output:
            continue
    
    print(output)
    output = [cat.split('. ')[1] for cat in output.split('\n')]
    return output
