import asyncio
import time

import pandas as pd
from dotenv import load_dotenv
from md2pdf.core import md2pdf
from ..apirequests import invoke_chute


def gen_report(
    theme: str,
    data: pd.DataFrame,
    token: str,
    metadata: dict,
    model_name: str = "deepseek-ai/DeepSeek-V3-0324",
    batch_size: int = 32000,
    are_problems: bool = False
):
    """Генерирует отчёт по теме (проблеме), если она является важной (не ошибочной)"""
    instr1 = (
        "Ты помощник в составлении отчетов о компании \"{company}\". "
        "Вот краткое описание компании: {desc}.\n\n"
        "Твоя задача - написать детализированный и аргументированный отчёт "
        "на заданную тему (проблему) по отзывам пользователей о компании, "
        "если она является важной и обоснованной, иначе, "
        "если это субъективное мнение, не подкреплённое конкретными примерами "
        "проблемы, верни "
        '"Заданная тема не является важной в контексте проблем бизнеса.".'
        "\nНе добавляй в конце отчёта свои комментарии."
        "\n\n**Цель:** Подготовить структурированный "
        "и детализированный подотчет на тему {theme}\n\n"
        "### Формат отчета и требования:\n- **Формат:** Markdown\n"
        "- **Заголовки:** Используй заголовки с уровня 2 (##).\n"
        "- **Анализ:** Выдели основные проблемы и предложения, "
        "которые пишут в своих отзывах пользователи.\n"
        "- **Стиль:** Стремись к аналитическому и связному изложению;\n\n"
        "Формат входа\n\n1. текст отзыва\n----\n2. текст отзыва\n----\n"
        "... (все оставшиеся отзывы)\n----\nn. текст отзыва\n\n"
        "Твой отчет:\n## {theme}"
    )
    instr2 = (
        "Ты помощник в составлении отчетово компании \"{company}\". "
        "Вот краткое описание компании: {desc}.\n\n"
        "Твоя задача - дополнить отчёт на заданную тему (проблему) "
        "из дополнительных отзывов пользователей.\n"
        "Не добавляй в конце отчёта свои комментарии.\n\n"
        "**Цель:** Дополнить структурированный и детализированный "
        "подотчет на тему {theme}\n\n"
        "### Формат отчета и требования:\n- **Формат:** Markdown\n"
        "- **Заголовки:** Используй заголовки с уровня 2 (##).\n"
        "- **Анализ:** Выдели основные проблемы и предложения, "
        "которые пишут в своих отзывах пользователи.\n"
        "- **Стиль:** Стремись к аналитическому и связному изложению;\n\n"
        "Формат входа\n\nТекст подотчёта\n\n--------\n\n(n+1). текст отзыва\n----\n"
        "(n+2). текст отзыва\n----\n... (все оставшиеся отзывы)\n----\n"
        "(n+k). текст отзыва"
    )
    instr1 = instr1.format(company=metadata['company'],
                           desc=metadata['description'], theme=theme)
    instr2 = instr2.format(company=metadata['company'],
                           desc=metadata['description'], theme=theme)
    
    df = data[["summary"]].reset_index(drop=True)
    df = df.dropna(how="all")
    df["len"] = df["summary"].str.len()
    df["cumlen"] = df["len"].cumsum()
    df["cumlen"] = df["cumlen"] + [6 * i for i in range(len(df))]

    zero_batch = df.loc[df["cumlen"] <= batch_size, "summary"]
    zero_batch = [str(j + 1) + ". " for j in range(len(zero_batch))] + zero_batch
    prompt = "\n----\n".join(zero_batch) + f"\n\nТвой отчет:\n## {theme}"

    output = ""
    while not output:
        try:
            time.sleep(10)
            output = asyncio.run(invoke_chute(prompt, instr1, token, model_name))
        except asyncio.exceptions.TimeoutError:
            continue
            # time.sleep(10)
            # output = asyncio.run(invoke_chute(prompt, model_name, instruction=instr1))

    if "Заданная тема не является важной" in output:
        print(output)
        return None

    prev_output = output
    i = 1
    start_index = len(zero_batch)
    while i * batch_size <= df.iloc[-1, -1]:
        batch = df.loc[
            (i * batch_size < df["cumlen"]) & (df["cumlen"] <= (i + 1) * batch_size),
            "summary",
        ]
        batch = [str(j + start_index) + ". " for j in range(len(batch))] + batch
        start_index = len(batch)
        prompt = prev_output + "\n\n--------\n\n" + "\n----\n".join(batch)

        time.sleep(10)
        try:
            output = asyncio.run(invoke_chute(prompt, instr2, token, model_name))
        except asyncio.exceptions.TimeoutError:
            continue

        if not output:
            continue

        prev_output = output
        i += 1

    print(output)

    if "```" in output:
        output = output.replace("```markdown", "").replace("```", "")

    if output.startswith("## "):
        output = output.split("\n\n", 1)[1]

    if "\n\n---\n" in output:
        output = output.rsplit("\n\n---\n", 1)[0]

    return output


def form_report(summaries: pd.DataFrame, clusters: pd.DataFrame,
                token: str, metadata: dict, save_to: str = "report.pdf",
                are_problems=False):
    """Формирует отчёт из подотчётов по каждому кластеру и конвертирует в pdf"""
    subreports = []
    for i in range(len(clusters)):
        time.sleep(10)
        subreports.append(
            gen_report(
                str(clusters.loc[i, "name"]),
                summaries[summaries["new_cluster"] == clusters.loc[i, "cluster"]],
                token, metadata, are_problems=are_problems
            )
        )

    reports = ""
    for i, line in clusters.iterrows():
        if subreports[i] is None:
            continue

        reports += f"""

## {line['name']}

##### {(summaries["new_cluster"] == clusters.loc[i, "cluster"]).sum()} отзывов

{subreports[i]}

"""
    md2pdf(
        save_to,
        md_content=f"""# Отчет по отзывам и сообщениям о компании "{company}"

{reports}
""",
        css_file_path="src/get_report/styles.css",  # Updated path
        base_url=".",
    )
