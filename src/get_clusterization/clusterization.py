import asyncio
import time
import warnings
from functools import partial

import numpy as np
import optuna
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import pairwise_distances
from torch import Tensor
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5 import T5EncoderModel

from ..apirequests import invoke_chute

warnings.filterwarnings("ignore")

CLUSTERING_ALGORITHMS = ["kmeans", "dbscan", "agg", "hdbscan", "optics"]
CLUSTERING_ALGORITHM_ALIASES = {
    "kmeans": "k-средних",
    "dbscan": "DBSCAN",
    "agg": "агломеративная кластеризация",
    "hdbscan": "HDBSCAN",
    "optics": "OPTICS",
}


def process_clustering_correction(output: str):
    """Обработка структурированного запроса для улучшения кластеризации"""
    output = output.split("\n\n", 1)[1]
    cluster_desc, *results = output.rsplit("\n\n", 3)
    cluster_problems = []
    for cluster in cluster_desc.split("Кластер ")[1:]:
        k, other = cluster.split(": ", 1)
        k = int(k)
        theme, other = other.strip().split("\n", 1)
        theme = theme.replace("[", "").replace("]", "")
        outliers = None
        if "Выбросы:" in other:
            problems, outliers = other.split("- Выбросы:")
            problems = problems.strip().split("Проблемы:")[1]

            outliers = outliers.strip()
            if outliers == "-":
                outliers = None
            else:
                outliers = [
                    outlier.split(". ", 1)[1] for outlier in outliers.split("\n")
                ]
        else:
            problems = other.split("Проблемы:\n")[1]

        problems = problems.strip().split("\n")
        if isinstance(problems, list):
            problems = [problem.strip().split(" ", 1)[1] for problem in problems]
        elif isinstance(problems, str):
            problems = [problems.split(" ", 1)[1]]

        cluster_problems.append(
            {"cluster": k, "theme": theme, "problems": problems, "outliers": outliers}
        )

    # print(cluster_problems)

    results = [results[i].split(":", 1)[1].strip() for i in range(len(results))]

    divide_clusters = None
    if results[0] != "-":
        results[0] = results[0].split("\n")
        if isinstance(results[0], list):
            results[0] = [
                [int(s) for s in results[0][i].split(" ") if s.isdigit()]
                for i in range(len(results[0]))
            ]
        else:
            results[0] = [[int(s) for s in results[0].split(" ") if s.isdigit()]]

        divide_clusters = {
            results[0][i][0]: results[0][i][1] for i in range(len(results[0]))
        }
        # print(divide_clusters)

    union_clusters = None
    if results[1] != "-":
        results[1] = results[1].split("\n")
        if isinstance(results[1], list):
            results[1] = [
                list(map(int, results[1][i].split(": ", 1)[1].split(", ")))
                for i in range(len(results[1]))
            ]
        elif isinstance(results[1], str):
            results[1] = [results[1].split(": ", 1)[1].split(", ")]

        union_clusters = results[1][:]
        # print(union_clusters)

    delete_clusters = (
        list(map(int, results[2].split(", "))) if results[2] != "-" else None
    )
    # print(delete_clusters)

    return cluster_problems, divide_clusters, union_clusters, delete_clusters


# ---- ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ТЕКСТА


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def base_embeds(data: list[str], model, tokenizer, max_length=4096):
    """Генерация эмбеддингов обычными моделями"""
    tokens = tokenizer(
        data, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**tokens)
    return last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])


def task_embeds(
    data: list[str],
    model,
    tokenizer,
    task: str = "paraphrase",
    max_length=512,
    pooling_method="mean",
):
    """Генерация эмбеддингов task моделями, с инструкциями для решения определённой задачи"""
    assert task in [
        "search_query",
        "paraphrase",
        "categorize",
        "categorize_sentiment",
        "categorize_topic",
        "categorize_entailment",
    ]
    assert pooling_method in ["mean", "cls"]

    data = [task + ": " + text for text in data]
    tokens = tokenizer(
        data, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**tokens)

    return pool(
        outputs.last_hidden_state,
        tokens["attention_mask"],
        pooling_method=pooling_method,
    )


def gen_embeddings(
    data: list[str],
    model_path: str = "ai-forever/FRIDA",
    task: str | None = None,
    cache_dir: str | None = None,
    normalize: bool = False,
):
    """Генерация эмбеддингов текста выбранной моделью"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if model_path == "ai-forever/FRIDA":
        model = T5EncoderModel.from_pretrained(model_path, cache_dir=cache_dir)
    else:
        model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)

    try:
        max_len = model.config.max_position_embeddings
    except AttributeError:
        max_len = (
            tokenizer.model_max_length if tokenizer.model_max_length <= 1 << 30 else 512
        )

    if task:
        embeddings = task_embeds(
            data,
            model,
            tokenizer,
            task,
            max_len,
            "cls" if model_path == "ai-forever/FRIDA" else "mean",
        )
    else:
        embeddings = base_embeds(data, model, tokenizer, max_len)

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


# ---- КЛАСТЕРИЗАЦИЯ


def is_large_data(X):
    """
    Определяет, считаются ли данные "большими" на основе доступной памяти.

    Параметры:
    ----------
    X : np.ndarray или pd.DataFrame
        Входные данные.
    mem_threshold_gb : float
        Порог (в ГБ), после которого данные считаются большими.

    Возвращает:
    -----------
    bool
        True, если данные большие, иначе False.
    """
    # is_large_data.mem_threshold_gb = 2
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    dtype_memory = X.nbytes / (X.shape[0] * X.shape[1])

    # Оценка памяти для матрицы расстояний (в ГБ)
    estimated_mem_usage_gb = (X.shape[0] ** 2) * dtype_memory / (1024**3)
    return (
        estimated_mem_usage_gb > is_large_data.mem_threshold_gb or available_mem_gb < 2
    )


is_large_data.mem_threshold_gb = 2


def compute_distance_matrix(X, metric="euclidean"):
    """
    Вычисляет матрицу расстояний, если данные не слишком большие.

    Параметры:
    ----------
    X : np.ndarray или pd.DataFrame
        Входные данные.
    metric : str
        Метрика расстояния ("euclidean", "cosine", "manhattan" и т.д.).

    Возвращает:
    -----------
    np.ndarray или None
        Матрица расстояний или None, если данные слишком большие.
    """
    if is_large_data(X):
        return None

    return pairwise_distances(X, metric=metric)


def get_param_distributions(algorithm_name, X):
    """
    Возвращает распределения параметров для Optuna.
    """
    n_samples, n_features = X.shape

    if algorithm_name == "kmeans":
        return {
            "n_clusters": optuna.distributions.IntUniformDistribution(
                2, min(20, n_samples // 2)
            ),
        }
    elif algorithm_name == "dbscan":
        return {
            "eps": optuna.distributions.FloatDistribution(0.001, 1.0, step=0.001),
            "min_samples": optuna.distributions.IntUniformDistribution(
                2, min(10, n_features * 2)
            ),
        }
    elif algorithm_name == "agg":
        return {
            "n_clusters": optuna.distributions.IntUniformDistribution(
                2, min(20, n_samples // 2)
            ),
            "linkage": optuna.distributions.CategoricalDistribution(
                ["ward", "complete", "average", "single"]
            ),
        }
    elif algorithm_name == "hdbscan":
        return {
            "cluster_selection_epsilon": optuna.distributions.FloatDistribution(
                0.001, 1.0, step=0.001
            ),
            "min_cluster_size": optuna.distributions.IntUniformDistribution(
                2, min(20, n_samples // 2)
            ),
        }
    elif algorithm_name == "optics":
        return {
            "min_samples": optuna.distributions.IntUniformDistribution(
                2, min(10, n_features * 2)
            ),
            "max_eps": optuna.distributions.FloatDistribution(0.001, 2.0, step=0.001),
            "cluster_method": optuna.distributions.CategoricalDistribution(
                ["xi", "dbscan"]
            ),
        }
    else:
        raise ValueError(f"Алгоритм {algorithm_name} не поддерживается.")


def cluster_with_algorithm(X, algorithm_name, params, metric="euclidean",
                           distance_matrix=None):
    """
    Запускает кластеризацию с заданными параметрами.

    Параметры:
    ----------
    X : np.ndarray
        Входные данные.
    algorithm_name : str
        Название алгоритма.
    params : dict
        Гиперпараметры.
    distance_matrix : np.ndarray или None
        Предвычисленная матрица расстояний.

    Возвращает:
    -----------
    np.ndarray
        Метки кластеров.
    """

    if algorithm_name == "kmeans":
        model = KMeans(**params)
    elif algorithm_name == "dbscan":
        model = DBSCAN(
            **params,
            metric="precomputed" if distance_matrix is not None else metric,
            n_jobs=-1
        )
    elif algorithm_name == "agg":
        model = AgglomerativeClustering(
            **params,
            metric="precomputed"
            if distance_matrix is not None and params['linkage'] != 'ward'
            else 'euclidean'
        )
    elif algorithm_name == "hdbscan":
        model = HDBSCAN(
            **params,
            metric="precomputed" if distance_matrix is not None else metric,
            n_jobs=-1
        )
    elif algorithm_name == "optics":
        model = OPTICS(
            **params,
            metric="precomputed" if distance_matrix is not None else metric,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Алгоритм {algorithm_name} не поддерживается.")

    if distance_matrix is not None and algorithm_name not in ["kmeans"]:
        try:
            labels = model.fit_predict(distance_matrix)
        except ValueError:
            labels = model.fit_predict(X)
    else:
        X_pos = X - X.min() if algorithm_name == "optics" else X.copy()
        labels = model.fit_predict(X_pos)

    return labels


def objective(
    trial: optuna.trial.Trial,
    X: np.ndarray,
    algorithm_name: str,
    metric="euclidean",
    distance_matrix: np.ndarray = None,
    use_silhouette: bool = True,
):
    """
    Целевая функция для Optuna с поддержкой всех типов распределений.
    """
    assert algorithm_name in CLUSTERING_ALGORITHMS

    params = {}
    param_distributions = get_param_distributions(algorithm_name, X)

    for param_name, distribution in param_distributions.items():
        if isinstance(distribution, optuna.distributions.IntUniformDistribution):
            params[param_name] = trial.suggest_int(
                param_name, distribution.low, distribution.high, step=distribution.step
            )
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            params[param_name] = trial.suggest_float(
                param_name, distribution.low, distribution.high, step=distribution.step
            )
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            params[param_name] = trial.suggest_categorical(
                param_name, distribution.choices
            )
        elif isinstance(distribution, optuna.distributions.LogUniformDistribution):
            params[param_name] = trial.suggest_float(
                param_name, distribution.low, distribution.high, log=True
            )
        elif isinstance(distribution, optuna.distributions.DiscreteUniformDistribution):
            params[param_name] = trial.suggest_float(
                param_name, distribution.low, distribution.high, step=distribution.q
            )
        else:
            raise ValueError(f"Неизвестный тип распределения: {type(distribution)}")

    labels = cluster_with_algorithm(X, algorithm_name, params, metric, distance_matrix)

    if len(np.unique(labels)) < 3:
        return -1.0 if use_silhouette else 0.0

    if use_silhouette:
        if distance_matrix is not None:
            return silhouette_score(distance_matrix, labels, metric="precomputed")
        else:
            return silhouette_score(X, labels, metric=metric)
    else:
        return calinski_harabasz_score(X, labels)


def optimize_clustering(
    X,
    algorithm_name="kmeans",
    metric="euclidean",
    use_silhouette=True,
    n_trials=50,
    n_jobs=-1,
):
    """
    Оптимизирует гиперпараметры кластеризации через Optuna.

    Параметры:
    ----------
    X : np.ndarray или pd.DataFrame
        Входные данные.
    algorithm_name : str
        Название алгоритма ("kmeans", "dbscan", "agg").
    metric : str
        Метрика расстояния ("euclidean", "cosine", "manhattan").
    use_silhouette : bool
        Оптимизировать silhouette_score (True) или calinski_harabasz_score (False).
    n_trials : int
        Количество испытаний Optuna.
    n_jobs : int
        Количество ядер для параллельных вычислений.

    Возвращает:
    -----------
    dict
        Лучшие параметры.
    np.ndarray
        Метки кластеров.
    """
    X = np.array(X)
    distance_matrix = (
        compute_distance_matrix(X, metric) if not is_large_data(X) else None
    )

    study = optuna.create_study(direction="maximize")
    objective_partial = partial(
        objective,
        X=X,
        algorithm_name=algorithm_name,
        metric=metric,
        distance_matrix=distance_matrix,
        use_silhouette=use_silhouette,
    )

    study.optimize(objective_partial, n_trials=n_trials, n_jobs=n_jobs)

    best_params = study.best_params
    best_labels = cluster_with_algorithm(
        X, algorithm_name, best_params, metric, distance_matrix
    )

    return best_params, best_labels, study.best_value


def clustering_selection(
    data: list[str],
    n_trials,
    save_folder="src/clustering/",
    *,
    embeddings_model: str = "ai-forever/FRIDA",
    cache_dir=None,
    normalize=False,
    task="paraphrase",
    large_data_thr=1,
    metric="euclidean",
    use_silhouette=True,
    n_jobs=-1,
    exclude_algorithms: list[str] = None
):
    """Подбирает наилучший алгоритм кластеризации по метрике"""
    embeds = gen_embeddings(data, embeddings_model, task=task, cache_dir=cache_dir,
                            normalize=normalize)

    is_large_data.mem_threshold_gb = large_data_thr
    best_results = {}
    best_score = -1 if use_silhouette else 0
    best_alg = None
    
    if exclude_algorithms is not None:
        alg_list = [alg for alg in CLUSTERING_ALGORITHMS
                    if alg not in exclude_algorithms]
    else:
        alg_list = CLUSTERING_ALGORITHMS
        
    for alg in alg_list:
        params, labels, score = optimize_clustering(
            embeds, alg, use_silhouette=use_silhouette, n_trials=n_trials,
            n_jobs=n_jobs, metric=metric
        )
        best_results[alg] = {"params": params, "labels": labels, "score": score}
        if score > best_score:
            best_score = score
            best_alg = alg

    # print(best_score, best_alg, best_results[best_alg]['params'])
    (
        pd.DataFrame(
            {"summary": data, "cluster": best_results[best_alg]["labels"]}
        ).to_csv(save_folder + "clustered_summaries1.csv")
    )
    return embeds, best_alg


def clusterization(
    data: list[str],
    n_trials,
    save_folder="src/clustering/",
    *,
    embeddings_model: str = "ai-forever/FRIDA",
    cache_dir=None,
    task="paraphrase",
    large_data_thr=1,
    use_silhouette=True,
    n_jobs=-1,
):
    embeds = gen_embeddings(data, embeddings_model, task=task, cache_dir=cache_dir)
    is_large_data.mem_threshold_gb = large_data_thr


# ---- ПРИМЕНЕНИЕ LLM ДЛЯ УЛУЧШЕНИЯ КЛАСТЕРИЗАЦИИ


def multilabel_classification(
    reviews: pd.DataFrame,
    categories: list,
    api_token: str,
    metadata: dict,
    model_name="deepseek-ai/DeepSeek-V3-0324",
    batch_size=24000
) -> list[dict[str, str | None]]:
    """Распределение проблем, упоминаемых в отзывах, на заданные категории и остальные"""
    df = reviews[['text']].reset_index(drop=True)
    df = df.dropna(how="all")
    df["len"] = df["text"].str.len()
    df["cumlen"] = df["len"].cumsum()
    df["cumlen"] = df["cumlen"] + [6 * i for i in range(len(df))]
    
    instr = (
        "Ты - опытный помощник по выявлению проблем компании "
        f"\"{metadata['company']}\", на которые жалуются клиенты в своих отзывах. "
        f"Вот краткое описание компании: {metadata['description']}.\n\n"
        "Твоя задача - максимально точно перечислить "
        "все конкретные проблемы и жалобы, упоминаемые пользователем, связанные "
        "с бизнесом, не теряя уточняющие детали. "
        "Каждую упоминаемую проблему отнеси к одному из предложенных классов: "
        f"{', '.join(categories)}, остальные - если нет проблем, "
        'которые относятся к классу, ставь символ "-", '
        "и если проблема не относится ни к одному классу, "
        'относи её к классу "остальные".'
        "Соблюдай шаблон ввода:\n1. текст отзыва\n----\n"
        "2. текст отзыва\n----\n... (все оставшиеся отзывы)\n----\n"
        "n. текст отзыва\n\nШаблон вывода:\n\n1.\n"
    )
    
    instr += "\n".join(
        [k + f': проблема1; проблема2, связанных с "{k}" в отзыве 1\n'
         for k in categories]
    )
    instr += (
        "\nостальные: перечисление проблем в первом отзыве, не относящихся "
        "ни к одному из классов выше\n\n"
        "... (все остальные отзывы)\n\nn.\n"
    )
    instr += "\n".join(
        [k + f': перечисление проблем, связанных с "{k}" ' f"в отзыве n\n"
         for k in categories]
    )
    instr += (
        "\nостальные: перечисление проблем в последнем отзыве, не относящихся "
        "ни к одному из классов выше"
    )
    
    i = 0
    outputs = []
    while i * batch_size <= df.iloc[-1, -1]:
        batch = df.loc[(i * batch_size < df['cumlen'])
                       & (df['cumlen'] < (i + 1) * batch_size), "text"]
        batch = [str(j + 1) + '. ' for j in range(len(batch))] + batch
        prompt = "\n----\n".join(batch)
        # print(instr)
        # print(prompt)
        
        time.sleep(10)
        try:
            if 'mistral' not in model_name:
                output = asyncio.run(
                    invoke_chute(prompt, instr, api_token, model_name)
                )
                if '</think>' in output:
                    output = output.split('</think>', 1)[1]
            else:
                output = asyncio.run(
                    invoke_mistral(prompt, instr, api_token, model_name)
                )
        
        except asyncio.exceptions.TimeoutError:
            continue
        
        if not output:
            continue
        
        outputs.append(output)
        i += 1
    
    outputs = [s.strip().split('.\n', 1)[1] if '.\n' in s
               else s.strip().split('. ', 1)[1]
               for part in outputs for s in part.split('\n\n')]
    
    problems = []
    for review in outputs:
        review = review
        review_categories = review.split('\n') if '\n' in review else [review]
        review_cats = dict.fromkeys(categories + ['остальные'], None)
        for category in review_categories:
            name, enum_problems = category.split(': ', 1)
            if name in review_cats:
                review_cats[name] = (enum_problems if enum_problems.strip() != '-'
                                     else None)
        
        problems.append(review_cats)
    
    return problems


def transform_cluster_labels(
    labels: np.ndarray,
    embeddings: Tensor,
    divide_clusters: dict[int, int] = None,
    union_clusters: list[list[int]] = None,
):
    """Преобразует метки кластеров в соответствии с заданными изменениями"""
    # Если пересечение разделяемых и объединяемых кластеров не пустое,
    # то исключаем в объединяемых все, что разделяются
    if not (divide_clusters is None or union_clusters is None) and set(
        divide_clusters.keys()
    ) & set(
        [
            union_clusters[i][j]
            for i in range(len(union_clusters))
            for j in range(len(union_clusters[i]))
        ]
    ):
        i = 0
        while i < len(union_clusters):
            for k in divide_clusters.keys():
                if k in union_clusters[i]:
                    union_clusters[i].remove(k)

            if len(union_clusters[i]) <= 1:
                del union_clusters[i]
                continue

            i += 1

    # Заменяем метки кластеров на новые через KMeans от заданного числа кластеров
    max_cluster = labels.max()
    if divide_clusters is not None:
        for i in divide_clusters.keys():
            indices = labels == i
            cluster_labels = KMeans(
                n_clusters=divide_clusters[i], n_init=20
            ).fit_predict(embeddings[indices])
            labels[indices] = cluster_labels + max_cluster + 1
            max_cluster = labels.max()

    # Заменяем метки кластеров меткой наименьшего кластера в группе
    if union_clusters and union_clusters[0]:
        for group in union_clusters:
            group_label = min(group)
            group.remove(group_label)
            for k in group:
                labels[labels == k] = group_label

    # Смещаем все метки до значений от 0 до числа кластеров - 1
    n_labels = np.unique(labels).size
    for k in range(n_labels):
        while k not in labels:
            labels[labels > k] -= 1

    return labels


def clustering_correction(
    clusters_folder: str,
    embeddings: Tensor,
    model_token,
    metadata: dict,
    model_name: str = "deepseek-ai/DeepSeek-V3-0324",
    best_alg: str = "kmeans",
    timeout=15,
):
    """Отправляет запросы к LLM для улучшения кластеризации
       и формирует их названия и обоснование"""
    assert best_alg in CLUSTERING_ALGORITHMS

    instr1 = (
        "Ты — аналитик отзывов. Проанализируй суммаризации негативных отзывов "
        f"о компании \"{metadata['company']}\", разбитые на кластеры, "
        "и выдели ключевые проблемы в каждом. "
        f"Вот краткое описание компании: {metadata['description']}.\n\n"
        "Задача:\n1. Выдели **ключевые проблемы**, "
        "которые чаще всего упоминаются в кластере.\n"
        "2. Укажи (если есть) выбросы (отзывы, не соответствующие общей "
        "теме кластера).\n"
        "3. В конце по выявленным проблемам укажи метки кластеров, "
        "которые можно было бы разъединить, и на какое количество кластеров, "
        'если такие есть, иначе поставь символ "-".\n'
        "4. Группы меток кластеров, которые можно было бы объединить из-за "
        "схожести их тематик (проблем), если такие есть, "
        'иначе поставь символ "-".'
        "5. Метки кластеров, которые можно удалить, если они не описывают"
        " проблем, связанных с бизнесом, если такие есть, "
        'иначе поставь символ "-".\nНЕ ПИШИ ПОЯСНЕНИЙ К ВЫБОРУ! '
        "НЕ ИСПОЛЗУЙ ОФОРМЛЕНИЕ MARKDOWN!"
        "\n\nИсходные данные:\n"
        "- Алгоритм кластеризации: {название_алгоритма} "
        "(например, K-Means, DBSCAN, иерархическая кластеризация).\n"
        "- Количество кластеров: {число_кластеров}.\n"
        "- Формат входа - тексты и метки их кластеров:\n\n"
        "Кластер 0:\nтекст1\n----\nтекст2\n----\n...\n\n"
        "Кластер 1:\nтекст1\n----\nтекст2\n----\n...\n\n...\n\n"
        "-Формат вывода:\n\nКластер N: тема кластера N\n - Проблемы:\n"
        "1. ... \n2. ... \n - Выбросы:\n1. текст отзыва\n"
        "2. текст отзыва\n...\n\nРазделённые кластеры:\n"
        "Кластер k1 на n1 кластеров\nКластер k2 на n2 кластеров\n\n"
        "Объединённые кластеры:\nГруппа кластеров 1: k1, k2, k3, ...\n"
        "Группа кластеров 2: k4, k5, ...\n\nЛишние кластеры: k8, k9, ..."
    )

    df = pd.read_csv(clusters_folder + "clustered_summaries1.csv", index_col=0)
    labels = df["cluster"].unique()
    labels.sort()
    prompt = (
        f"- Алгоритм кластеризации: {CLUSTERING_ALGORITHM_ALIASES[best_alg]}.\n"
        f"- Количество кластеров: {len(labels)}.\n\n"
    )
    for k in labels:
        prompt += f"Кластер {k}:\n"
        cluster = df.loc[df["cluster"] == k, ["summary"]]
        # cluster['len'] = cluster['summary'].str.len()
        cluster['cumlen'] = cluster['summary'].str.len().cumsum()
        cluster = cluster[cluster['cumlen'] <= 30000 / len(labels)]

        prompt += "\n----\n".join(cluster['summary'].to_list())
        prompt += "\n\n"

    # print(prompt)
    while True:
        try:
            output = asyncio.run(
                invoke_chute(
                    prompt, token=model_token, model=model_name, instruction=instr1
                )
            )
        except asyncio.exceptions.TimeoutError:
            continue
        
        if not output:
            continue
        
        if "</think>" in output:
            _, output = output.split("</think>\n", 1)
        
        print(output)
        try:
            _, divide_clusters, union_clusters, delete_clusters = (
                process_clustering_correction(output)
            )
        except IndexError:
            time.sleep(10)
            continue
        
        break

    if delete_clusters is not None:
        embeds = embeddings[~df["cluster"].isin(delete_clusters)]
        df = df[~df["cluster"].isin(delete_clusters)].reset_index(drop=True)
    else:
        embeds = embeddings

    new_labels = transform_cluster_labels(
        df["cluster"].to_numpy().copy(), embeds, divide_clusters, union_clusters
    )

    df["new_cluster"] = new_labels
    df.to_csv(clusters_folder + "clustered_summaries2.csv")

    instr2 = (
        "Ты помощник в составлении отчетов. Твоя задача из списка отзывов "
        f"о компании \"{metadata['company']}\" ниже выявить одну общую категорию, "
        "к которой они относятся. Вот краткое описание компании: "
        f"{metadata['description']}.\n\nФормат входа - отзывы из одного кластера:"
        "\n\nотзыв1\n----\nотзыв2\n----\n...\n\nФормат вывода:\n\n"
        "Название категории: [название общей категории сообщений из кластера]"
        "\nОбоснование: [рассуждения почему ты выбрал именно эту категорию]"
    )

    labels = df["new_cluster"].unique()
    labels.sort()
    categories = []
    for k in labels:
        cluster = df.loc[df["new_cluster"] == k, ["summary"]]
        cluster['cumlen'] = cluster['summary'].str.len().cumsum()
        cluster = cluster[cluster['cumlen'] < 30000]

        prompt = "\n----\n".join(cluster['summary'].to_list())
        output = ""
        while not output:
            try:
                output = asyncio.run(
                    invoke_chute(prompt, instr2, model_token, model=model_name)
                )
            except asyncio.exceptions.TimeoutError:
                continue
             
            if "</think>" in output:
                _, output = output.split("</think>\n", 1)
            
        print(output, end='\n\n')
        name, reasoning = output.split("\n", 1)
        name = (
            name.split(": ", 1)[1].replace("**", "")
                .replace("[", "").replace("]", "")
        )
        reasoning = reasoning.split(": ", 1)[1]
        categories.append({"cluster": k, "name": name, "reasoning": reasoning})
        time.sleep(timeout)

    categories = pd.DataFrame(categories)
    categories.to_csv(clusters_folder + "categories.csv")
    return categories


def MasterClusterization(
    data,
    chutes_token,
    n_trials=200,
    save_folder="",
    embeddings_model="ai-forever/FRIDA",
    cache_dir=None,
    large_data_thr=1,
    use_silhouette=True,
    n_jobs=-1,
) -> pd.DataFrame:
    embeds, best_alg = clustering_selection(
        data["summary"].to_list().copy(),
        n_trials,
        save_folder,
        embeddings_model=embeddings_model,
        cache_dir=cache_dir,
        large_data_thr=large_data_thr,
        use_silhouette=use_silhouette,
        n_jobs=n_jobs,
    )
    res = clustering_correction(
        save_folder, embeds, model_token=chutes_token, best_alg=best_alg
    )
    return res
