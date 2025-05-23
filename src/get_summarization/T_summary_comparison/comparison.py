import pandas as pd
import asyncio
import time

from ...apirequests import invoke_chute, invoke_mistral


def summary_comparison(instr2, token):
    parsers_path = "src/get_info/parsers/"

    gm = pd.read_csv(parsers_path + "google_maps/google_reviews.csv")
    ot = pd.read_csv(parsers_path + "otzovik/otzovik_reviews.csv")
    tg = pd.read_csv(parsers_path + "telegram/mriya_messages.csv")
    tg = tg[tg["rating"] == 2]
    df = pd.concat([gm, ot, tg], ignore_index=True)

    df = df.dropna(how="all")
    df["len"] = df["text"].str.len()
    df["cumlen"] = df["len"].cumsum()
    df["cumlen"] = df["cumlen"] + [8 * i for i in range(len(df))]
    max_texts = len(df[df["cumlen"] < 100000])
    df = df[:max_texts]
    prompt = "\n\n----\n\n".join(df["text"])
    # print(prompt)

    # output = asyncio.run(invoke_chute(prompt))
    # output = asyncio.run(invoke_mistral(prompt))

    compare_models = [
        "deepseek-ai/DeepSeek-V3-0324",
        "Qwen/Qwen3-235B-A22B",
        "mistral-small-latest",
    ]
    f = open("output_examples3.txt", "w", encoding="utf-8")
    f.write(" | ".join(compare_models) + "\n\n")

    # instr2 = (
    #     "Ты - опытный помощник по выявлению проблем бизнеса, на которые жалуются "
    #     "клиенты в своих отзывах. Твоя задача - максимально точно перечислить "
    #     "все конкретные проблемы и жалобы, упоминаемые пользователем, связанные "
    #     "с бизнесом, не теряя уточняющие детали. "
    #     "Каждую упоминаемую проблему отнеси к одному из предложенных классов: "
    #     "столовая, номер, мероприятия, персонал, остальные - если нет проблем, "
    #     'которые относятся к классу, ставь символ "-", '
    #     "и если проблема не относится ни к одному классу, "
    #     'относи её к классу "остальные".'
    #     "Соблюдай шаблон ввода:\n\n1. текст отзыва\n\n----\n\n"
    #     "2. текст отзыва\n\n----\n\n... (все оставшиеся отзывы)\n\n----\n\n"
    #     "n. текст отзыва\n\nШаблон вывода:\n\n1.\n"
    # )

    # instr2 += "\n".join(
    #     [
    #         k + f': перечисление проблем с разделителем ";", '
    #         f'связанных с "{k}" в отзыве 1\n'
    #         for k in "столовая, номер, мероприятия, персонал".split(", ")
    #     ]
    # )
    # instr2 += (
    #     "\nостальные: перечисление проблем в первом отзыве, не относящихся "
    #     "ни к одному из классов выше\n\n----\n\n"
    #     "... (все остальные отзывы)\n\n----\n\nn.\n"
    # )
    # instr2 += "\n".join(
    #     [
    #         k + f': перечисление проблем, связанных с "{k}" ' f"в последнем отзыве\n"
    #         for k in "столовая, номер, мероприятия, персонал".split(", ")
    #     ]
    # )
    # instr2 += (
    #     "\nостальные: перечисление проблем в последнем отзыве, не относящихся "
    #     "ни к одному из классов выше"
    # )

    outputs = []
    for model_name in compare_models:
        output = ""
        print(model_name)
        i = 0
        while len(output.split("----")) != max_texts:
            time.sleep(15)
            print(i := i + 1)
            print(output)
            if "mistral" not in model_name:
                output = asyncio.run(invoke_chute(prompt, instr2, token, model_name))
                if "</think>" in output:
                    _, output = output.split("</think>\n", 1)
            else:
                output = asyncio.run(invoke_mistral(prompt, instr2, token, model_name))

        outputs.append([s.strip() for s in output.split("----")])

    for i in range(max_texts):
        f.write("\n\n----\n\n".join([outputs[j][i] for j in range(len(outputs))]))
        f.write("\n\n------------\n\n")

    f.close()
