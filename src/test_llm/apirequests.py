import json
import os

import aiohttp
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
        "max_tokens": 32000,
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
                       "\n----\nтекст отзыва пользователя\n----\n"
                       "... (все оставшиеся отзывы)\n----\nтекст отзыва\n\n"
                       "и шаблон вывода:\n\nсуммаризация первого отзыва"
                       "\n----\nсуммаризация второго отзыва\n----\n"
                       "... (суммаризация всех оставшихся отзывов)\n----\n"
                       "суммаризация последнего отзыва")

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
