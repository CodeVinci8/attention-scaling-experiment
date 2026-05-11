from part_a_numpy_attention import (
    run_numpy_experiment,
    generate_example_attention_weights,
)
from plotting import (
    plot_heatmap,
    plot_max_weight_vs_dk,
    plot_entropy_vs_dk,
)


def main() -> None:
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


if __name__ == "__main__":
    main()