import asyncio
import json
import os

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


async def invoke_chute(query, model="deepseek-ai/DeepSeek-V3-0324", role="user",
                       instruction=None):
    api_token = os.environ.get("CHUTES_API_TOKEN")
    
    headers = {
        "Authorization": "Bearer " + api_token,
        "Content-Type": "application/json"
    }
    
    if instruction is None:
        instruction = ("Ты - опытный помощник по выявлению проблем бизнеса, "
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
                       "\n\nсуммаризация последнего отзыва")
    
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": instruction
            },
            {
                "role": role,
                "content": query
            }
        ],
        "stream": True,
        "max_tokens": 16000,
        "temperature": 0.6
    }
    
    output = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://llm.chutes.ai/v1/chat/completions",
            headers=headers,
            json=body
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = data.strip()
                        if chunk == 'None':
                            continue
                            
                        chunk = json.loads(chunk)
                        if chunk['choices'][0]['delta']['content']:
                            output += chunk['choices'][0]['delta']['content']
                            # print(chunk['choices'][0]['delta']['content'], end='')
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
    
    return output


async def invoke_mistral(query, model="mistral-small-latest", role="user", instruction=None):
    api_key = os.environ["MISTRAL_API_TOKEN"]
    client = Mistral(api_key=api_key)
    
    if instruction is None:
        instruction = ("Ты - опытный помощник по выявлению проблем бизнеса, "
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
                       "\n\nсуммаризация последнего отзыва")

    response = await client.chat.stream_async(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instruction
            },
            {
                "role": role,
                "content": query
            }
        ],
    )
    
    output = ""
    async for chunk in response:
        if chunk.data.choices[0].delta.content is not None:
            output += chunk.data.choices[0].delta.content
            # print(chunk.data.choices[0].delta.content, end="")
    
    return output

gm = pd.read_csv("C:\Code\Social-scan\get_info\google_maps\google_reviews.csv")
ot = pd.read_csv("C:\Code\Social-scan\get_info\otzovik\otzovik_reviews.csv")
tg = pd.read_csv("C:\Code\Social-scan\get_info\\telegram\\mriya_messages.csv")
tg = tg[tg['rating'] == 2]
df = pd.concat([gm, ot, tg], ignore_index=True)

df = df.dropna(how='all')
df['len'] = df['text'].str.len()
df['cumlen'] = df['len'].cumsum()
df['cumlen'] = df['cumlen'] + [8*i for i in range(len(df))]
max_texts = len(df[df['cumlen'] < 100000])
df = df[:max_texts]
prompt = "\n\n----\n\n".join(df['text'])
# print(prompt)

# # Qwen/Qwen3-235B-A22B
# output = asyncio.run(invoke_chute(prompt))
# output = asyncio.run(invoke_mistral(prompt))

compare_models = ["deepseek-ai/DeepSeek-V3-0324", "Qwen/Qwen3-235B-A22B",
                  "mistral-small-latest"]
f = open('output_examples2.txt', 'w', encoding='utf-8')
f.write(" | ".join(compare_models) + '\n\n')

instr2 = ("Ты - опытный помощник по выявлению проблем бизнеса, на которые жалуются "
          "клиенты в своих отзывах. Твоя задача - максимально точно перечислить "
          "все конкретные проблемы и жалобы, упоминаемые пользователем, связанные "
          "с бизнесом, не теряя уточняющие детали. Соблюдай шаблон ввода:"
          "\n\nтекст отзыва пользователя\n\n----\n\nтекст отзыва пользователя"
          "\n\n----\n\n... (все оставшиеся отзывы)\n\n----\n\nтекст отзыва\n\n"
          "и шаблон вывода:\n\n")



outputs = []
for model_name in compare_models:
    output = ""
    print(model_name)
    i = 0
    while len(output.split('----')) != max_texts:
        print(i := i + 1)
        print(output)
        if "mistral" not in model_name:
            output = asyncio.run(invoke_chute(prompt, model_name))
            if '</think>' in output:
                _, output = output.split('</think>\n', 1)
        else:
            output = asyncio.run(invoke_mistral(prompt, model_name))
    
    outputs.append([s.strip() for s in output.split('----')])

for i in range(max_texts):
    for j in range(len(outputs)):
        f.write(outputs[j][i] + '\n\n')
    
    f.write('--------\n\n')

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
