import asyncio
import json
import os

import aiohttp
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


async def invoke_chute(query, model="deepseek-ai/DeepSeek-V3-0324", role="user"):
    api_token = os.environ.get("CHUTES_API_TOKEN")
    
    headers = {
        "Authorization": "Bearer " + api_token,
        "Content-Type": "application/json"
    }
    
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Ты - опытный помощник по сокращению объёма текста "
                           "и умеешь точно выделать из него всю суть. "
                           "Твоя задача - максимально точно передать главную "
                           "причину недовольства пользователя использованием. "
                           "Сокращай объём текста минимум до 128 символов. "
                           "Соблюдай шаблон входа:\n\n"
                           "текст отзыва пользователя\n\n----\n\n"
                           "текст отзыва пользователя\n\n----\n\n"
                           "... (все оставшиеся отзывы)\n\n----\n\nтекст отзыва\n\n"
                           "и шаблон вывода:\n\nсуммаризация первого отзыва\n\n"
                           "----\n\nсуммаризация второго отзыва\n\n----\n\n"
                           "... (суммаризация всех оставшихся отзывов)\n\n----"
                           "\n\nсуммаризация последнего отзыва отзыва"
            },
            {
                "role": role,
                "content": query
            }
        ],
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.7
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
                        if chunk:
                            output += chunk['choices'][0]['delta']['content']
                            print(chunk['choices'][0]['delta']['content'], end='')
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
    
    return output


gm = pd.read_csv("C:\Code\Social-scan\get_info\google_maps\google_reviews.csv")
ot = pd.read_csv("C:\Code\Social-scan\get_info\otzovik\otzovik_reviews.csv")
tg = pd.read_csv("C:\Code\Social-scan\get_info\\telegram\\mriya_messages.csv")
tg = tg[tg['rating'] == 2]
df = pd.concat([gm, ot, tg], ignore_index=True)

df = df.dropna(how='all')
df['len'] = df['text'].str.len()
df['cumlen'] = df['len'].cumsum()
df['cumlen'] = df['cumlen'] + [6*i for i in range(len(df))]
prompt = "\n\n----\n\n".join(df.loc[:len(df[df['cumlen'] < 32000]), 'text'])
# print(prompt)

output = asyncio.run(invoke_chute(prompt))
df['summary'] = [s.strip() for s in output.split('----')]
print(df['summary'])
# Qwen/Qwen3-235B-A22B
# chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8
