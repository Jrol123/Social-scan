import asyncio
import time

from pandas import DataFrame

from ...apirequests import invoke_chute, invoke_mistral

NNAME = {
    "mistral": [invoke_mistral, "mistral-small-latest"],
    "chute": [invoke_chute, "deepseek-ai/DeepSeek-V3-0324"],
    "deepseek": [invoke_chute, "deepseek-ai/DeepSeek-V3-0324"],
}
"""
Возможные варианты для использования нейросетей.
"""

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


def gen_summarization(
    reviews: DataFrame,
    token: str,
    model_name: str = "mistral",
    instr=DEFAULT_INSTRUCTION,
    batch_size: int = 32000,
) -> list[str]:
    """
    Генерация суммаризаций для каждого отзыва.

    Args:
        reviews (DataFrame): DataFrame с отзывами.
        model_name (str): Имя модели.
        token (str): Токен для модели.
        instr (str, optional): Инструкция по суммаризации. Defaults to DEFAULT_INSTRUCTION.
        batch_size (int, optional): Размер батча. Defaults to 32000.

    Raises:
        ValueError: Неправильное имя модели.

    Returns:
        list[str]: Список суммаризаций по каждому отзыву.
    """
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
    reviews: DataFrame,
    token: str,
    model_name: str,
    metadata: dict,
    batch_size: int = 32000,
) -> list[str]:
    """
    Генерация категорий для суммаризации

    Args:
        reviews (DataFrame): DataFrame с отзывами.
        model_name (str): Имя модели.
        token (str): Токен для модели.
        metadata (dict): _description_
        batch_size (int, optional): _description_. Defaults to 32000.

    Raises:
        ValueError: Неправильное имя модели.

    Returns:
        list[str]: Список категорий.
    """
    instr1 = (
        f"Ты - аналитик отзывов о компании \"{metadata['company']}\"."
        f"Вот краткое описание компании: {metadata['description']}.\n\n"
        "Проанализируй следующие пользовательские отзывы и выдели основные "
        "категории проблем, которые в них упоминаются, учитывая специфику "
        "области компании.\n\nСоблюдай шаблон ввода:\n\nтекст отзыва 1"
        "\n----\nтекст отзыва 2\n----\n... (все оставшиеся отзывы)\n----\n"
        "текст отзыва n\n\nШаблон вывода:\n\n1. Категория 1\n2. Категория 2"
        "\n...\nk. Категория k"
    )

    # TODO: metadata сразу вставлять в инструкцию в Config
    
    df = reviews[["text"]].reset_index(drop=True)
    df = df.dropna(how="all")
    df["len"] = df["text"].str.len()
    df["cumlen"] = df["len"].cumsum()
    df["cumlen"] = df["cumlen"] + [6 * i for i in range(len(df))]
    
    print()
    
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
    output = [
        category.split(". ", 1)[1].split("\n", 1)[0].replace("**", "").replace(":", "")
        for category in output
        if category[0].isdigit()
    ]

    instr2 = (
        f"Ты - аналитик отзывов о компании \"{metadata['company']}\"."
        f"Вот краткое описание компании: {metadata['description']}.\n\n"
        "Выдели непересекающиеся категории, сформированные по отзывам.\n\n"
        "Соблюдай шаблон ввода:\n\n1. Категория 1\n2. Категория 2"
        "\n...\nk. Категория k"
        "Шаблон вывода:\n\n1. Категория 1\n2. Категория 2"
        "\n...\nk. Категория k-n"
    )

    prompt = "\n".join([str(i + 1) + ". " + cat for i, cat in enumerate(output)])
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

        if not output or '\n\n' in output:
            continue

        break

    output = output.strip()
    print(output)
    try:
        output = [
            cat.split(". ", 1)[1].strip()

            for cat in (output.split("\n") if '\n' in output else [output])
        ]
    except Exception:
        return gen_categories(reviews, token, model_name, metadata, batch_size)
    
    output = [cat.replace('**', '') if '**' in cat else cat for cat in output]
    output = [cat.split('(', 1)[0] if '(' in cat else cat for cat in output]
    return output


def gen_multilabel_summarization(
    reviews: DataFrame,
    categories: list,
    metadata: dict,
    token: str,
    model_name="deepseek",
    batch_size=24000,
) -> list[dict[str, str | None]]:
    """Распределение проблем, упоминаемых в отзывах, на заданные категории и остальные"""
    df = reviews[["text"]]
    df = df
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
        [
            k + f': проблема1; проблема2, связанных с "{k}" в отзыве 1\n'
            for k in categories
        ]
    )
    instr += (
        "\nостальные: перечисление проблем в первом отзыве, не относящихся "
        "ни к одному из классов выше\n\n"
        "... (все остальные отзывы)\n\nn.\n"
    )
    instr += "\n".join(
        [
            k + f': перечисление проблем, связанных с "{k}" ' f"в отзыве n\n"
            for k in categories
        ]
    )
    instr += (
        "\nостальные: перечисление проблем в последнем отзыве, не относящихся "
        "ни к одному из классов выше"
    )

    i = 0
    outputs = []
    while i * batch_size <= df.iloc[-1, -1]:
        batch = df.loc[
            (i * batch_size < df["cumlen"]) & (df["cumlen"] < (i + 1) * batch_size),
            "text",
        ]
        batch = [str(j + 1) + ". " for j in range(len(batch))] + batch
        prompt = "\n----\n".join(batch)
        # print(instr)
        # print(prompt)

        time.sleep(10)
        try:
            if model_name in NNAME.keys():
                output = asyncio.run(
                    NNAME[model_name][0](
                        query=prompt,
                        instruction=instr,
                        token=token,
                        model=NNAME[model_name][1],
                    )
                )
                if model_name in ["chute", "deepseek"]:
                    if "</think>" in output:
                        output = output.split("</think>", 1)[1]
            else:
                raise ValueError("Неправильное имя модели!")

        except asyncio.exceptions.TimeoutError:
            continue

        if (not output or not ('.\n' in output or '. ' in output)
           or '\n\n' not in output or '----' in output
           or output.strip().count('\n\n') != len(batch) - 1):
            continue
        
        try:
            outputs.extend([
                s.strip().split(".\n", 1)[1]
                if ".\n" in s else s.strip().split(". ", 1)[1]
                for s in output.split("\n\n")
            ] if '\n\n' in output else [output])
        except Exception:
            continue
            
        i += 1
    
    problems = []
    for review in outputs:
        review = review
        review_categories = review.split("\n") if "\n" in review else [review]
        review_categories = [r.strip().replace('**', '')
                             for r in review_categories if r.strip()]
        review_cats = dict.fromkeys(categories + ["остальные"], None)
        print(review_categories)
        for category in review_categories:
            if ': ' not in category:
                continue
                
            name, enum_problems = category.split(": ", 1)
            if name in review_cats:
                review_cats[name] = (
                    enum_problems
                    if not enum_problems.strip().startswith("-")
                    else None
                )

        problems.append(review_cats)

    return problems
