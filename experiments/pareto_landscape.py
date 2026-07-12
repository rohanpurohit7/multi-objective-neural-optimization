"""Visualize a non-convex, two-objective Pareto problem."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def objectives(x: np.ndarray) -> np.ndarray:
    """Return two competing, non-convex objectives for candidate x values."""
    f1 = (x + 1.1) ** 2 + 0.22 * np.sin(7.0 * x)
    f2 = (x - 1.1) ** 2 + 0.22 * np.cos(6.0 * x + 0.3)
    return np.column_stack((f1, f2))


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Mark non-dominated rows for a minimization problem."""
    keep = np.ones(len(values), dtype=bool)
    for i, candidate in enumerate(values):
        dominated = np.all(values <= candidate, axis=1) & np.any(values < candidate, axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def scalarized_indices(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Find the best sampled candidate for each weight on objective one."""
    scores = weights[:, None] * values[:, 0] + (1 - weights[:, None]) * values[:, 1]
    return np.argmin(scores, axis=1)


def main() -> None:
    x = np.linspace(-2.5, 2.5, 1200)
    values = objectives(x)
    frontier = pareto_mask(values)
    weights = np.linspace(0.0, 1.0, 17)
    selected = scalarized_indices(values, weights)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(x, values[:, 0], label="Objective 1: accuracy proxy")
    axes[0].plot(x, values[:, 1], label="Objective 2: cost proxy")
    axes[0].set(title="Non-convex objectives", xlabel="Decision parameter x", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].scatter(values[:, 0], values[:, 1], s=8, alpha=0.22, label="Candidates")
    order = np.argsort(values[frontier, 0])
    pf = values[frontier][order]
    axes[1].plot(pf[:, 0], pf[:, 1], color="#d62728", lw=3, label="Sampled Pareto frontier")
    axes[1].scatter(values[selected, 0], values[selected, 1], marker="x", s=55,
                    color="#111111", label="Weighted-sum choices")
    axes[1].set(title="Objective space", xlabel="Objective 1", ylabel="Objective 2")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("One optimum becomes a frontier when objectives conflict", fontsize=15)
    fig.tight_layout()
    output = Path(__file__).resolve().parents[1] / "artifacts" / "pareto_landscape.svg"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
