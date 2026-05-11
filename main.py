import argparse

from part_a_numpy_attention import (
    run_numpy_experiment,
    generate_example_attention_weights,
)
from part_b_tiny_transformer import run_transformer_experiment
from plotting import (
    plot_heatmap,
    plot_max_weight_vs_dk,
    plot_entropy_vs_dk,
    plot_loss_comparison,
)


def run_part_a() -> None:
    print("Part A: NumPy attention experiment started...")

    df = run_numpy_experiment()

    print("\nTable 1 results:")
    print(df.round(4))

    weights_unscaled, weights_scaled = generate_example_attention_weights(d_k=64)

    plot_heatmap(
        weights_unscaled,
        "Attention weights without scaling",
        "fig2_heatmap_unscaled.png",
    )

    plot_heatmap(
        weights_scaled,
        "Attention weights with scaling",
        "fig3_heatmap_scaled.png",
    )

    plot_max_weight_vs_dk(df)
    plot_entropy_vs_dk(df)

    print("\nPart A completed.")
    print("Files saved to outputs/:")
    print("- table1_attention_metrics.csv")
    print("- fig2_heatmap_unscaled.png")
    print("- fig3_heatmap_scaled.png")
    print("- fig4_max_weight_vs_dk.png")
    print("- fig4_extra_entropy_vs_dk.png")


def run_part_b() -> None:
    print("Part B: Tiny Transformer experiment started...")

    history_scaled, history_unscaled, table2 = run_transformer_experiment()

    plot_loss_comparison(
        history_scaled=history_scaled,
        history_unscaled=history_unscaled,
        loss_column="train_loss",
        ylabel="training loss",
        title="Training loss comparison",
        filename="fig5_train_loss.png",
    )

    plot_loss_comparison(
        history_scaled=history_scaled,
        history_unscaled=history_unscaled,
        loss_column="val_loss",
        ylabel="validation loss",
        title="Validation loss comparison",
        filename="fig6_val_loss.png",
    )

    print("\nPart B completed.")
    print("Files saved to outputs/:")
    print("- history_scaled_transformer.csv")
    print("- history_unscaled_transformer.csv")
    print("- table2_training_metrics.csv")
    print("- fig5_train_loss.png")
    print("- fig6_val_loss.png")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--part",
        choices=["a", "b", "all"],
        default="a",
        help="Which part of the experiment to run",
    )

    args = parser.parse_args()

    if args.part in ["a", "all"]:
        run_part_a()

    if args.part in ["b", "all"]:
        run_part_b()


if __name__ == "__main__":
    main()