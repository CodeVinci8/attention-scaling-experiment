import math
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Устойчивая softmax-функция.

    Вычитание максимума защищает от переполнения exp().
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_attention_weights(
    q: np.ndarray,
    k: np.ndarray,
    scaled: bool = True,
) -> np.ndarray:
    """
    Считает attention-веса.

    q: матрица запросов Q, shape = (num_tokens, d_k)
    k: матрица ключей K, shape = (num_tokens, d_k)
    scaled: если True, делим QK^T на sqrt(d_k)

    Возвращает:
    attention weights, shape = (num_tokens, num_tokens)
    """
    d_k = q.shape[-1]

    scores = q @ k.T

    if scaled:
        scores = scores / math.sqrt(d_k)

    weights = softmax(scores, axis=-1)
    return weights


def mean_max_weight(weights: np.ndarray) -> float:
    """
    Средний максимальный вес внимания по строкам.
    Чем ближе к 1, тем резче attention.
    """
    return float(weights.max(axis=-1).mean())


def mean_entropy(weights: np.ndarray) -> float:
    """
    Средняя энтропия attention-распределения по строкам.
    Чем выше энтропия, тем распределение ровнее.
    """
    eps = 1e-12
    entropy = -np.sum(weights * np.log(weights + eps), axis=-1)
    return float(entropy.mean())

