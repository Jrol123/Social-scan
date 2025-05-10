import pandas as pd


df = pd.read_csv("parsed_data.csv", index_col=0)
df['rating'] = [line['rating'] if not pd.isna(line['rating']) else
                line['stars'] if not pd.isna(line['stars']) else None
                for i, line in df[['rating', 'stars']].iterrows()]
df = df.drop('stars', axis=1)
df = df[['service_id', 'name', 'additional_id', 'date', 'rating', 'text', 'answer']]
print(df)
