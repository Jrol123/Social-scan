"""
Этот пример посвящён кластеризации, третьему этапу в pipeline.

Здесь происходит суммаризация отзывов.
"""
from dotenv import dotenv_values
from pandas import read_csv
from src.get_clusterization import MasterClusterization

if __name__ == "__main__":
    data = read_csv("examples/example_summarized_data.csv", index_col=0)
    data = read_csv("examples/example_summarized_data.csv", index_col=0)
    # print(*data[:10], sep='\n\n')

    secrets = dotenv_values()

    MasterClusterization(
        data,
        secrets["CHUTES_API_TOKEN"],
        100,
        "examples/04_clusterization/",
        embeddings_model="ai-forever/FRIDA",
        # cache_dir="D:/TRANSFORMERS_MODELS",
        large_data_thr=1,
        use_silhouette=True,
        n_jobs=-1,
    )