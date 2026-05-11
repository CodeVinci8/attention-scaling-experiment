import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


OUTPUT_DIR = "outputs"


def plot_heatmap(weights: np.ndarray, title: str, filename: str) -> None:
    """
    Строит heatmap attention-весов.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(weights, aspect="auto")
    plt.colorbar(label="attention weight")

    plt.xlabel("Key token index")
    plt.ylabel("Query token index")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def plot_max_weight_vs_dk(df: pd.DataFrame) -> None:
    """
    Строит график зависимости среднего максимального attention-веса от d_k.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(7, 5))

    plt.plot(
        df["d_k"],
        df["max_without_scaling"],
        marker="o",
        label="without scaling",
    )

    plt.plot(
        df["d_k"],
        df["max_with_scaling"],
        marker="o",
        label="with scaling",
    )

    plt.xlabel("d_k")
    plt.ylabel("mean max attention weight")
    plt.title("Influence of d_k on attention sharpness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_max_weight_vs_dk.png"), dpi=300)
    plt.close()


def plot_entropy_vs_dk(df: pd.DataFrame) -> None:
    """
    Дополнительный полезный график: энтропия attention от d_k.
    Его можно использовать в статье вместо одного из слабых рисунков
    или добавить как дополнительный материал.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(7, 5))

    plt.plot(
        df["d_k"],
        df["entropy_without_scaling"],
        marker="o",
        label="without scaling",
    )

    plt.plot(
        df["d_k"],
        df["entropy_with_scaling"],
        marker="o",
        label="with scaling",
    )

    plt.xlabel("d_k")
    plt.ylabel("mean entropy")
    plt.title("Influence of d_k on attention entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_extra_entropy_vs_dk.png"), dpi=300)
    plt.close()