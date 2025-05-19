import json
import os

import aiohttp
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

CHUTE_MODELS = ["deepseek-ai/DeepSeek-V3-0324"]
MISTRAL_MODELS = ["mistral-small-latest"]


async def invoke_chute(
    query, instruction, token, model="deepseek-ai/DeepSeek-V3-0324", role="user"
):
    # api_token = os.environ.get("CHUTES_API_TOKEN")
    # if not api_token:
    #     raise ValueError(
    #         "CHUTES_API_TOKEN is missing. Please set it in the environment variables."
    #     )
    
    if model not in CHUTE_MODELS:
        raise ValueError("Неправильное имя модели!")

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": role, "content": query},
        ],
        "stream": True,
        "max_tokens": 32000,
        "temperature": 0.6,
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
    query, instruction, token, model="mistral-small-latest", role="user"
):
    # api_token = os.environ["MISTRAL_API_TOKEN"]
    # if not api_token:
    #     raise ValueError(
    #         "MISTRAL_API_TOKEN is missing. Please set it in the environment variables."
    #     )
    
    if model not in MISTRAL_MODELS:
        raise ValueError("Неправильное имя модели!")
    
    client = Mistral(api_key=token)

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
