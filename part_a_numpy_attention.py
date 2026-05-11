import os
import numpy as np
import pandas as pd

from utils import compute_attention_weights, mean_entropy, mean_max_weight


OUTPUT_DIR = "outputs"

D_K_VALUES = [8, 16, 32, 64, 128, 256]
NUM_TOKENS = 10
NUM_TRIALS = 500
RANDOM_SEED = 42


def run_numpy_experiment() -> pd.DataFrame:
    """
    Проводит эксперимент для разных d_k.

    Для каждого d_k:
    1. генерирует случайные Q и K;
    2. считает attention без scaling;
    3. считает attention со scaling;
    4. измеряет max weight и entropy;
    5. сохраняет таблицу результатов.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for d_k in D_K_VALUES:
        max_unscaled_values = []
        entropy_unscaled_values = []
        max_scaled_values = []
        entropy_scaled_values = []

        for _ in range(NUM_TRIALS):
            q = rng.normal(loc=0.0, scale=1.0, size=(NUM_TOKENS, d_k))
            k = rng.normal(loc=0.0, scale=1.0, size=(NUM_TOKENS, d_k))

            weights_unscaled = compute_attention_weights(q, k, scaled=False)
            weights_scaled = compute_attention_weights(q, k, scaled=True)

            max_unscaled_values.append(mean_max_weight(weights_unscaled))
            entropy_unscaled_values.append(mean_entropy(weights_unscaled))

            max_scaled_values.append(mean_max_weight(weights_scaled))
            entropy_scaled_values.append(mean_entropy(weights_scaled))

        row = {
            "d_k": d_k,
            "max_without_scaling": np.mean(max_unscaled_values),
            "entropy_without_scaling": np.mean(entropy_unscaled_values),
            "max_with_scaling": np.mean(max_scaled_values),
            "entropy_with_scaling": np.mean(entropy_scaled_values),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = os.path.join(OUTPUT_DIR, "table1_attention_metrics.csv")
    df.to_csv(output_path, index=False)

    return df


def generate_example_attention_weights(d_k: int = 64):
    """
    Генерирует один пример attention-весов для heatmap.
    Важно: Q и K одинаковые для scaled и unscaled сравнения.
    """
    rng = np.random.default_rng(123)

    q = rng.normal(loc=0.0, scale=1.0, size=(NUM_TOKENS, d_k))
    k = rng.normal(loc=0.0, scale=1.0, size=(NUM_TOKENS, d_k))

    weights_unscaled = compute_attention_weights(q, k, scaled=False)
    weights_scaled = compute_attention_weights(q, k, scaled=True)

    return weights_unscaled, weights_scaled