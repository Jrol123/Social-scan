import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from sentiment_analysis import MasterSentimentAnalysis


def check_tokens_capacity(df, model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Ошибка загрузки модели {model_name}: {str(e)}")
        raise e
    
    print(model_name, '|', tokenizer.model_max_length, end=' ')
    mxlen = tokenizer.model_max_length
    try:
        print(model.config.max_position_embeddings, end=' ')
        mxlen = min(mxlen, model.config.max_position_embeddings)
    except:
        pass
    
    try:
        print(model.config.n_positions)
        mxlen = min(mxlen, model.config.n_positions)
    except:
        print()
    
    print()
    
    tokenized_text = [tokenizer(tx, return_tensors="pt")["input_ids"].shape[1]
                      for tx in df['text']]
    df['tokens'] = tokenized_text
    print(df['tokens'].quantile(0.95), sum(df['tokens'] <= mxlen) / len(df), end='\n\n')


df = pd.read_csv("parsed_data.csv", index_col=0)

model_name = "sismetanin/mbart_ru_sum_gazeta-ru-sentiment-rusentiment"
check_tokens_capacity(df, model_name)
# sent_an = MasterSentimentAnalysis(model_name, 1024, 12,
#                                   "binary", device='cpu')
# labeled_messages = sent_an.predict(df[df['service_id'].isin([3, 4])])
# labeled_messages = labeled_messages[labeled_messages['label'] == 1]
# df = pd.concat([df[~df['service_id'].isin([3, 4])],
#                 labeled_messages.drop('label', axis=1)], ignore_index=True)
# df = df.dropna(how='all')
# df.to_csv("filtered_data.csv")
