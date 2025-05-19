import asyncio
import pandas as pd

from ..apirequests import invoke_chute, invoke_mistral

NNAME = {
    "mistral": [invoke_mistral, "mistral-small-latest"],
    "chute": [invoke_chute, "deepseek-ai/DeepSeek-V3-0324"],
    "deepseek": [invoke_chute, "deepseek-ai/DeepSeek-V3-0324"],
}


def summarize_reviews(
    reviews: pd.DataFrame,
    model_name: str,
    token,
    instr,
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
