"""Visualize non-convexity, Pareto trade-offs, and confidence without model absolutism."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pareto_mask(values: np.ndarray) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for i, candidate in enumerate(values):
        dominated = np.all(values <= candidate, axis=1) & np.any(values < candidate, axis=1)
        keep[i] = not np.any(dominated)
    return keep


def main() -> None:
    rng = np.random.default_rng(7)

    # Panel A: a non-convex scalarized landscape over two decision parameters.
    x = np.linspace(-2.7, 2.7, 220)
    y = np.linspace(-2.7, 2.7, 220)
    xx, yy = np.meshgrid(x, y)
    loss = (
        0.10 * (xx**2 + yy**2)
        + 0.35 * np.sin(2.8 * xx) * np.cos(2.4 * yy)
        + 0.18 * np.sin(4.2 * (xx + yy))
        + 0.08 * (xx - yy) ** 2
    )

    # Panels B/C: three plausible cybersecurity models evaluated repeatedly.
    models = {
        "Sequence": ((0.39, 0.13), (0.045, 0.022), "#4C78A8"),
        "Transformer": ((0.24, 0.20), (0.034, 0.030), "#E45756"),
        "Vector augmented": ((0.17, 0.29), (0.028, 0.038), "#54A24B"),
    }
    samples = {}
    for name, (mean, sd, color) in models.items():
        draw = rng.normal(mean, sd, size=(700, 2))
        samples[name] = (np.clip(draw, 0.001, 0.999), color)

    means = np.array([draw.mean(axis=0) for draw, _ in samples.values()])
    efficient = pareto_mask(means)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    contour = axes[0].contourf(xx, yy, loss, levels=28, cmap="viridis")
    axes[0].contour(xx, yy, loss, levels=12, colors="white", alpha=0.28, linewidths=0.7)
    axes[0].scatter([-1.56, 0.25, 1.58], [-0.48, 1.36, -1.12], marker="x", s=90,
                    color="white", linewidth=2, label="Different local basins")
    axes[0].set(title="A. Non-convex decision landscape", xlabel="Parameter $\\theta_1$",
                ylabel="Parameter $\\theta_2$")
    axes[0].legend(loc="upper right")
    fig.colorbar(contour, ax=axes[0], fraction=0.046, label="Scalarized loss")

    for i, (name, (draw, color)) in enumerate(samples.items()):
        mean = draw.mean(axis=0)
        lo, hi = np.quantile(draw, [0.025, 0.975], axis=0)
        axes[1].errorbar(mean[0], mean[1], xerr=[[mean[0] - lo[0]], [hi[0] - mean[0]]],
                         yerr=[[mean[1] - lo[1]], [hi[1] - mean[1]]], fmt="o", ms=9,
                         color=color, capsize=4, label=name)
        axes[1].annotate("Pareto-efficient" if efficient[i] else "Dominated", mean,
                         xytext=(7, 7), textcoords="offset points", fontsize=8)

    # Equal-utility lines: slope changes with stakeholder preference.
    grid = np.linspace(0.08, 0.48, 100)
    for weight, style in [(0.25, "--"), (0.50, "-."), (0.75, ":")]:
        # w*miss + (1-w)*false_positive = constant
        anchor = means[np.argmin(weight * means[:, 0] + (1 - weight) * means[:, 1])]
        constant = weight * anchor[0] + (1 - weight) * anchor[1]
        line = (constant - weight * grid) / (1 - weight)
        axes[1].plot(grid, line, style, color="#333333", alpha=0.65,
                     label=f"Preference hyperplane w={weight:.2f}")
    axes[1].set(xlim=(0.08, 0.48), ylim=(0.07, 0.38), title="B. Hyperplanes select trade-offs",
                xlabel="Miss rate (security risk)", ylabel="False-positive rate (SOC workload)")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)

    weights = np.linspace(0, 1, 101)
    win_probability = {name: np.zeros_like(weights) for name in models}
    names = list(models)
    # Pair bootstrap draw k across models; winner minimizes weighted operational loss.
    for j, weight in enumerate(weights):
        utility = np.column_stack([
            weight * samples[name][0][:, 0] + (1 - weight) * samples[name][0][:, 1]
            for name in names
        ])
        winners = np.argmin(utility, axis=1)
        for i, name in enumerate(names):
            win_probability[name][j] = np.mean(winners == i)
    for name in names:
        axes[2].plot(weights, win_probability[name], lw=2.5, color=models[name][2], label=name)
    axes[2].axhline(0.95, color="#555555", ls="--", lw=1, label="95% confidence reference")
    axes[2].set(title="C. Confidence depends on priorities", xlabel="Weight on missed attacks",
                ylabel="Probability model minimizes weighted loss", ylim=(-0.02, 1.02))
    axes[2].grid(alpha=0.22)
    axes[2].legend(fontsize=8)

    fig.suptitle("Confidence describes a preference-conditioned solution—not a universally best model",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             "Higher security weight can favor richer context; higher workload weight can favor a simpler model. "
             "Overlapping intervals preserve decision uncertainty.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

    output = Path(__file__).resolve().parents[1] / "artifacts" / "confidence_tradeoff_hyperplane.svg"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

