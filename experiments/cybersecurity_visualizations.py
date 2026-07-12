"""Generate committed, browser-renderable figures for the cybersecurity lab."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n_pos = pos.sum()
    n_neg = len(y) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def curve(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    rows = []
    for threshold in np.linspace(np.quantile(score, .02), np.quantile(score, .995), 180):
        pred = score >= threshold
        tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
        fp = np.sum(pred & (y == 0)); tn = np.sum(~pred & (y == 0))
        rows.append((fp / (fp + tn), fn / (tp + fn), pred.mean()))
    return np.asarray(rows)


def main() -> None:
    rng = np.random.default_rng(42)
    n = 5000
    failed = rng.poisson(1.2, n)
    velocity = rng.gamma(2.0, 1.1, n)
    novelty = rng.beta(2.0, 5.0, n)
    travel = rng.binomial(1, 0.07, n)
    similarity = rng.beta(1.3, 5.0, n)
    latent = (0.55 * failed + 0.38 * velocity + 2.4 * novelty + 2.1 * travel
              + 3.0 * similarity + 1.3 * travel * novelty
              + 0.45 * np.sin(2.5 * velocity) + rng.normal(0, .8, n))
    labels = (latent > np.quantile(latent, .91)).astype(int)

    normalize = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    sequence = normalize(.60 * failed + .40 * velocity)
    vector = normalize(.38 * failed + .25 * velocity + 1.25 * novelty + 1.15 * travel + 1.45 * similarity)
    models = {"Sequence only": sequence, "Vector augmented": vector}
    colors = {"Sequence only": "#4C78A8", "Vector augmented": "#F58518"}
    curves = {name: curve(labels, score) for name, score in models.items()}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for name, rows in curves.items():
        axes[0].plot(rows[:, 0], rows[:, 1], lw=2.4, color=colors[name], label=name)
        axes[1].plot(rows[:, 0], 1 - rows[:, 1], lw=2.4, color=colors[name],
                     label=f"{name} (AUC={rank_auc(labels, models[name]):.3f})")
        axes[2].plot(rows[:, 2], rows[:, 1], lw=2.4, color=colors[name], label=name)
    axes[0].set(title="Pareto operating frontier", xlabel="False-positive rate", ylabel="Miss rate")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", alpha=.6)
    axes[1].set(title="Detection quality", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[2].set(title="Security versus SOC workload", xlabel="Alert rate", ylabel="Miss rate")
    for ax in axes:
        ax.grid(alpha=.22); ax.legend(fontsize=8)
    fig.suptitle("Vector context changes the attainable cybersecurity trade-off", fontsize=15, fontweight="bold")
    fig.tight_layout()
    root = Path(__file__).resolve().parents[1] / "artifacts"
    root.mkdir(exist_ok=True)
    first = root / "cybersecurity_operating_tradeoffs.svg"
    fig.savefig(first, format="svg", bbox_inches="tight")

    boot = []
    for _ in range(400):
        idx = rng.integers(0, n, n)
        if np.unique(labels[idx]).size == 2:
            boot.append(rank_auc(labels[idx], vector[idx]) - rank_auc(labels[idx], sequence[idx]))
    ci = np.quantile(boot, [.025, .975])
    budget = .10
    miss = {}
    for name, score in models.items():
        pred = score >= np.quantile(score, 1 - budget)
        miss[name] = np.sum((labels == 1) & ~pred) / np.sum(labels == 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].hist(boot, bins=30, color="#54A24B", alpha=.88)
    axes[0].axvline(0, color="black", ls="--")
    axes[0].axvspan(ci[0], ci[1], color="#ECA82C", alpha=.25, label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    axes[0].set(title="Confidence in AUC improvement", xlabel="Vector AUC − sequence AUC", ylabel="Bootstrap samples")
    axes[0].legend()
    axes[1].bar(list(miss), list(miss.values()), color=[colors[k] for k in miss])
    axes[1].set(title="Same 10% analyst-alert budget", ylabel="Miss rate", ylim=(0, .75))
    for i, value in enumerate(miss.values()):
        axes[1].text(i, value + .02, f"{value:.1%}", ha="center", fontweight="bold")
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.suptitle("Evidence supports improvement, not universal model superiority", fontsize=14, fontweight="bold")
    fig.tight_layout()
    second = root / "cybersecurity_confidence.svg"
    fig.savefig(second, format="svg", bbox_inches="tight")
    print(f"Wrote {first}\nWrote {second}")


if __name__ == "__main__":
    main()

