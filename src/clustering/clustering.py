import seaborn as sns
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(
            batch_size, device=last_hidden_states.device), sequence_lengths]


def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def base_embeds(data, model, tokenizer, max_length=4096):
    tokens = tokenizer(data, max_length=max_length, padding=True, truncation=True,
                       return_tensors="pt")
    outputs = model(**tokens)
    return last_token_pool(outputs.last_hidden_state, tokens['attention_mask'])


def task_embeds(data, model, tokenizer, task='paraphrase', max_length=512,
                pooling_method="mean"):
    assert task in ['search_query', 'paraphrase', 'categorize',
                    'categorize_sentiment', 'categorize_topic',
                    'categorize_entailment']
    assert pooling_method in ["mean", "cls"]
    
    data = [task + ': ' + text for text in data]
    tokens = tokenizer(data, max_length=max_length, padding=True, truncation=True,
                       return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    
    return pool(outputs.last_hidden_state, tokens["attention_mask"],
                pooling_method=pooling_method)



data = open('../test_llm/output_examples4.txt', encoding='utf-8').read()
data = data.strip().split('\n\n------------\n\n')
data = [ans.split('\n\n----\n\n')[2].split('. ', 1)[1] for ans in data]
print(*data[:10], sep='\n\n')
