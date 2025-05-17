import json
import os

import aiohttp
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

DEFAULT_INSTRUCTION = (
    "Ты - опытный помощник по выявлению проблем бизнеса, "
    "на которые жалуются клиенты в своих отзывах. "
    "Твоя задача - максимально точно перечислить все конкретные "
    "проблемы и жалобы, упоминаемые пользователем, связанные "
    "с бизнесом, не теряя уточняющие детали. "
    "Сокращай объём текста минимум до 256 символов. "
    "Соблюдай шаблон ввода:\n\nтекст отзыва пользователя"
    "\n\n----\n\nтекст отзыва пользователя\n\n----\n\n"
    "... (все оставшиеся отзывы)\n\n----\n\nтекст отзыва\n\n"
    "и шаблон вывода:\n\nсуммаризация первого отзыва\n\n"
    "----\n\nсуммаризация второго отзыва\n\n----\n\n"
    "... (суммаризация всех оставшихся отзывов)\n\n----"
    "\n\nсуммаризация последнего отзыва"
)


async def invoke_chute(
    query, model="deepseek-ai/DeepSeek-V3-0324", role="user", instruction=None
):
    api_token = os.environ.get("CHUTES_API_TOKEN")
    if not api_token:
        raise ValueError("CHUTES_API_TOKEN is missing. Please set it in the environment variables.")

    headers = {
        "Authorization": "Bearer " + api_token,
        "Content-Type": "application/json",
    }

    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": role, "content": query},
        ],
        "stream": True,
        "max_tokens": 32000,
        "temperature": 0.6

    }

    output = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://llm.chutes.ai/v1/chat/completions", headers=headers, json=body
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = data.strip()
                        if chunk == "None":
                            continue

                        chunk = json.loads(chunk)
                        if chunk["choices"][0]["delta"]["content"]:
                            output += chunk["choices"][0]["delta"]["content"]
                            # print(chunk['choices'][0]['delta']['content'], end='')
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")

    return output


async def invoke_mistral(
    query, model="mistral-small-latest", role="user", instruction=None
):
    api_key = os.environ["MISTRAL_API_TOKEN"]
    client = Mistral(api_key=api_key)

    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    response = await client.chat.stream_async(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": role, "content": query},
        ],
    )

    output = ""
    async for chunk in response:
        if chunk.data.choices[0].delta.content is not None:
            output += chunk.data.choices[0].delta.content
            # print(chunk.data.choices[0].delta.content, end="")

    return output


def process_deepseek_clustering_correction(output):
    output = output.split('Анализ кластеров:\n\n', 1)[1]
    cluster_desc, results = output.split('\n\n---\n\n', 1)
    cluster_problems = []
    for cluster in cluster_desc.split('#### **')[1:]:
        k, other = cluster.split(': ', 1)
        k = int(k.split()[1])
        theme, other = other.split('**\n', 1)
        outliers = None
        if '**Выбросы:**' in other:
            problems, outliers = other.split('\n- **Выбросы:**')
            problems = problems.split('**Проблемы:**\n')[1]
            outliers = outliers.split('\n  - *"')[1:]
            outliers = [outlier.rsplit('"*', 1)[0] for outlier in outliers]
        else:
            problems = other.split('**Проблемы:**\n')[1]
        
        problems = problems.strip().split('\n')
        if isinstance(problems, list):
            problems = [problem.strip().split(' ', 1)[1]
                        for problem in problems]
        elif isinstance(problems, str):
            problems = [problems.split(' ', 1)[1]]
        
        cluster_problems.append({
            'cluster': k,
            'theme': theme,
            'problems': problems,
            'outliers': outliers
        })
    
    print(cluster_problems)
    
    results = results.rsplit('**Итог:**\n', 1)[1]
    results = results.split('- **')[1:]
    results = [results[i].split('**', 1)[1].strip()
               for i in range(len(results))]
    
    results[0] = results[0].split('\n')
    results[0] = [[int(s) for s in results[0][i].split(' ') if s.isdigit()]
                  for i in range(len(results[0]))]
    divide_clusters = {results[0][i][0]: results[0][i][1]
                       for i in range(len(results[0]))}
    print(divide_clusters)
    
    results[1] = results[1].split('\n')
    if isinstance(results[1], list):
        results[1] = [list(map(int, results[1][i].split(': ', 1)[1].split(', ')))
                      for i in range(len(results[1]))]
    elif isinstance(results[1], str):
        results[1] = [results[1].split(': ', 1)[1].split(', ')]
    
    union_clusters = results[1][:]
    print(union_clusters)
    
    delete_clusters = list(map(int, results[2].split(', ')))
    print(delete_clusters)
    
    return cluster_problems, divide_clusters, union_clusters, delete_clusters


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

instr2 = (
    "Ты - опытный помощник по выявлению проблем бизнеса, на которые жалуются "
    "клиенты в своих отзывах. Твоя задача - максимально точно перечислить "
    "все конкретные проблемы и жалобы, упоминаемые пользователем, связанные "
    "с бизнесом, не теряя уточняющие детали. "
    "Каждую упоминаемую проблему отнеси к одному из предложенных классов: "
    "столовая, номер, мероприятия, персонал, остальные - если нет проблем, "
    'которые относятся к классу, ставь символ "-", '
    "и если проблема не относится ни к одному классу, "
    'относи её к классу "остальные".'
    "Сокращай текст до 128 символов на класс. "
    "Соблюдай шаблон ввода:"
    "\n\nтекст отзыва пользователя\n\n----\n\nтекст отзыва пользователя"
    "\n\n----\n\n... (все оставшиеся отзывы)\n\n----\n\nтекст отзыва\n\n"
    "и шаблон вывода:\n\n"
)


instr2 += "\n".join(
    [
        k + f': перечисление проблем, связанных с "{k}" ' f"в первом отзыве\n"
        for k in "столовая, номер, мероприятия, персонал".split(", ")
    ]
)
instr2 += (
    "\nостальные: перечисление проблем в первом отзыве, не относящихся "
    "ни к одному из классов выше\n\n----\n\n"
    "... (все остальные отзывы)\n\n----\n\n"
)
instr2 += "\n".join(
    [
        k + f': перечисление проблем, связанных с "{k}" ' f"в последнем отзыве\n"
        for k in "столовая, номер, мероприятия, персонал".split(", ")
    ]
)
instr2 += (
    "\nостальные: перечисление проблем в последнем отзыве, не относящихся "
    "ни к одному из классов выше"
)


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
            output = asyncio.run(invoke_chute(prompt, model_name, instruction=instr2))
            if "</think>" in output:
                _, output = output.split("</think>\n", 1)
        else:
            output = asyncio.run(invoke_mistral(prompt, model_name, instruction=instr2))

    outputs.append([s.strip() for s in output.split("----")])

for i in range(max_texts):
    f.write("\n\n----\n\n".join([outputs[j][i] for j in range(len(outputs))]))
    f.write("\n\n------------\n\n")

f.close()

# try:
#     df['summary'] = [s.strip() for s in output.split('----')]
#     for i, line in df[['text', 'summary']].iterrows():
#         print(line['text'], end="\n\n----\n\n")
#         print(line['summary'], end="\n\n----------\n\n")
# except Exception:
#     print(output)
#
# df = df.drop(['len', 'cumlen'], axis=1)
# df.to_csv("summarized_data.csv", index=False)
# print(df['summary'])
