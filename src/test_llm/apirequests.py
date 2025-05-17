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
