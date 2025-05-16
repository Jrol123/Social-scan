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
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

warnings.filterwarnings("ignore")

CLUSTERING_ALGORITHMS = ["kmeans", "dbscan", "agg", "hdbscan", "optics"]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
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

def gen_embeddings(data,
                   model_path: str = "ai-forever/FRIDA",
                   task: str | None = None,
                   cache_dir: str | None = None,
                   normalize: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if model_path == "ai-forever/FRIDA":
        model = T5EncoderModel.from_pretrained(model_path, cache_dir=cache_dir)
    else:
        model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
    
    try:
        max_len = model.config.max_position_embeddings
    except AttributeError:
        max_len = (tokenizer.model_max_length
                   if tokenizer.model_max_length <= 2**20 else 512)
    
    if task:
        embeddings = task_embeds(
            data, model, tokenizer, task, max_len,
            "cls" if model_path == "ai-forever/FRIDA" else "mean"
        )
    else:
        embeddings = base_embeds(
            data, model, tokenizer, max_len
        )
    
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


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
    available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
    dtype_memory =  X.nbytes / (X.shape[0] * X.shape[1])
    
    # Оценка памяти для матрицы расстояний (в ГБ)
    estimated_mem_usage_gb = (X.shape[0] ** 2) * dtype_memory / (1024 ** 3)
    return (estimated_mem_usage_gb > is_large_data.mem_threshold_gb
            or available_mem_gb < 2)

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
    Возвращает распределения параметров для Optuna (совместимые с v3.0+).
    """
    n_samples, n_features = X.shape
    
    if algorithm_name == "kmeans":
        return {
            "n_clusters": optuna.distributions.IntUniformDistribution(2, min(20,
                                                                             n_samples // 2)),
        }
    elif algorithm_name == "dbscan":
        return {
            "eps": optuna.distributions.FloatDistribution(0.001, 1.0, step=0.001),
            "min_samples": optuna.distributions.IntUniformDistribution(2, min(10,
                                                                              n_features * 2)),
        }
    elif algorithm_name == "agg":
        return {
            "n_clusters": optuna.distributions.IntUniformDistribution(2, min(20,
                                                                             n_samples // 2)),
            "linkage": optuna.distributions.CategoricalDistribution(
                ["ward", "complete", "average", "single"]),
        }
    elif algorithm_name == "hdbscan":
        return {
            "cluster_selection_epsilon": optuna.distributions.FloatDistribution(
                0.001, 1.0, step=0.001),
            "min_cluster_size": optuna.distributions.IntUniformDistribution(2,
                                                                            min(20,
                                                                                n_samples // 2)),
        }
    elif algorithm_name == "optics":
        return {
            "min_samples": optuna.distributions.IntUniformDistribution(2, min(10,
                                                                              n_features * 2)),
            "max_eps": optuna.distributions.FloatDistribution(0.001, 2.0,
                                                              step=0.001),
            "cluster_method": optuna.distributions.CategoricalDistribution(
                ["xi", "dbscan"]),
        }
    else:
        raise ValueError(f"Алгоритм {algorithm_name} не поддерживается.")

def cluster_with_algorithm(X, algorithm_name, params, distance_matrix=None):
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
        model = KMeans(**params, random_state=42)
    elif algorithm_name == "dbscan":
        model = DBSCAN(**params,
                       metric="precomputed"
                       if distance_matrix is not None else "euclidean")
    elif algorithm_name == "agg":
        model = AgglomerativeClustering(**params)
    elif algorithm_name == 'hdbscan':
        model = HDBSCAN(**params,
                        metric="precomputed"
                        if distance_matrix is not None else "euclidean")
    elif algorithm_name == 'optics':
        model = OPTICS(**params,
                       metric="precomputed"
                       if distance_matrix is not None else "euclidean",
                       n_jobs=-1)
    else:
        raise ValueError(f"Алгоритм {algorithm_name} не поддерживается.")
    
    if distance_matrix is not None and algorithm_name in ["dbscan", "hdbscan",
                                                          "optics"]:
        labels = model.fit_predict(distance_matrix)
    else:
        X_pos = X - X.min() if algorithm_name == 'optics' else X.copy()
        labels = model.fit_predict(X_pos)
    
    return labels

def objective(trial, X, algorithm_name, distance_matrix=None, use_silhouette=True):
    """
    Целевая функция для Optuna с поддержкой всех типов распределений.
    """
    params = {}
    param_distributions = get_param_distributions(algorithm_name, X)
    
    for param_name, distribution in param_distributions.items():
        if isinstance(distribution, optuna.distributions.IntUniformDistribution):
            params[param_name] = trial.suggest_int(
                param_name,
                distribution.low,
                distribution.high,
                step=distribution.step
            )
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            params[param_name] = trial.suggest_float(
                param_name,
                distribution.low,
                distribution.high,
                step=distribution.step
            )
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            params[param_name] = trial.suggest_categorical(
                param_name,
                distribution.choices
            )
        elif isinstance(distribution, optuna.distributions.LogUniformDistribution):
            params[param_name] = trial.suggest_float(
                param_name,
                distribution.low,
                distribution.high,
                log=True
            )
        elif isinstance(distribution,
                        optuna.distributions.DiscreteUniformDistribution):
            params[param_name] = trial.suggest_float(
                param_name,
                distribution.low,
                distribution.high,
                step=distribution.q
            )
        else:
            raise ValueError(f"Неизвестный тип распределения: {type(distribution)}")
    
    labels = cluster_with_algorithm(X, algorithm_name, params, distance_matrix)
    
    if len(np.unique(labels)) <= 3:
        return -1.0 if use_silhouette else 0.0
    
    if use_silhouette:
        if distance_matrix is not None:
            return silhouette_score(distance_matrix, labels, metric="precomputed")
        else:
            return silhouette_score(X, labels)
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
    distance_matrix = compute_distance_matrix(X, metric) if not is_large_data(
        X) else None
    
    study = optuna.create_study(direction="maximize")
    objective_partial = partial(
        objective,
        X=X,
        algorithm_name=algorithm_name,
        distance_matrix=distance_matrix,
        use_silhouette=use_silhouette,
    )
    
    study.optimize(objective_partial, n_trials=n_trials, n_jobs=n_jobs)
    
    best_params = study.best_params
    best_labels = cluster_with_algorithm(
        X, algorithm_name, best_params, distance_matrix
    )
    
    return best_params, best_labels, study.best_value


if __name__ == "__main__":
    data = open('../test_llm/output_examples4.txt', encoding='utf-8').read()
    data = data.strip().split('\n\n------------\n\n')
    data = [ans.split('\n\n----\n\n')[2].split('. ', 1)[1] for ans in data]
    # print(*data[:10], sep='\n\n')
    
    embeds = gen_embeddings(data, task="paraphrase")
    
    is_large_data.mem_threshold_gb = 1
    best_results = {}
    best_score = -1
    best_alg = None
    for alg in CLUSTERING_ALGORITHMS:
        params, labels, score = optimize_clustering(
            embeds, alg, use_silhouette=True, n_trials=100, n_jobs=-1
        )
        best_results[alg] = {'params': params, 'labels': labels, 'score': score}
        if score > best_score:
            best_score = score
            best_alg = alg
    
    print(best_score, best_alg, best_results[best_alg]['params'])
    (pd.DataFrame({'summary': data, 'cluster': best_results[best_alg]['labels']})
     .to_csv('clustered_summaries1.csv'))
    