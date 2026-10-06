from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata

POST_STEPS = 50
METHODS = ("no_augmentation", "random", "pauli", "gram", "oracle")


def make_figure(
    checkpoints: list[dict[str, object]],
    candidates: list[dict[str, object]],
    methods: list[dict[str, object]],
    traces: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.1,
            "figure.dpi": 180,
        }
    )
    colors = {
        "no_augmentation": "#777777",
        "random": "#e69f00",
        "pauli": "#377eb8",
        "gram": "#c44e7d",
        "oracle": "#222222",
    }
    differing = [
        row for row in checkpoints if int(row["pauli_gram_selected_different"])
    ]
    labels = {
        "no_augmentation": "no augmentation",
        "random": "random",
        "pauli": "candidate-only",
        "gram": "incremental",
        "oracle": "oracle",
    }
    representative_row = (
        differing
        or sorted(checkpoints, key=lambda item: (int(item["n"]), int(item["seed"])))
    )[0]
    representative = str(representative_row["instance"])
    ground = float(representative_row["ground_energy"])
    fig, axes = plt.subplots(2, 2, figsize=(8.25, 6.2), constrained_layout=True)
    steps = np.arange(POST_STEPS + 1)
    for method in METHODS:
        values = traces[f"{representative}__{method}"] - ground
        axes[0, 0].plot(
            steps,
            values,
            color=colors[method],
            linewidth=1.8 if method in ("gram", "oracle") else 1.25,
            linestyle="--" if method == "oracle" else "-",
            label=labels[method],
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Post-checkpoint optimization step")
    axes[0, 0].set_ylabel("Energy error $E-E_0$")
    axes[0, 0].set_title("(a) Representative same-point recovery")
    axes[0, 0].legend(frameon=False, loc="lower left")
    completed_sizes = sorted({int(row["n"]) for row in methods})
    plot_methods = ("no_augmentation", "random", "pauli", "gram")
    width = 0.18
    for method_index, method in enumerate(plot_methods):
        for n_index, n in enumerate(completed_sizes):
            values = np.asarray(
                [
                    float(row["gain_50"])
                    for row in methods
                    if row["method"] == method and int(row["n"]) == n
                ]
            )
            center = n_index + (method_index - 1.5) * width
            axes[0, 1].bar(
                center,
                float(np.mean(values)),
                width=0.9 * width,
                color=colors[method],
                alpha=0.72,
                label=labels[method] if n_index == 0 else None,
            )
            jitter = np.linspace(-0.04, 0.04, len(values))
            axes[0, 1].scatter(
                center + jitter,
                values,
                s=14,
                facecolor="white",
                edgecolor=colors[method],
                linewidth=0.8,
                zorder=3,
            )
    axes[0, 1].set_xticks(
        range(len(completed_sizes)), [f"n={n}" for n in completed_sizes]
    )
    axes[0, 1].set_ylabel("50-step energy decrease")
    axes[0, 1].set_title("(b) Five fixed seeds per completed size")
    lower, upper = axes[0, 1].get_ylim()
    axes[0, 1].set_ylim(lower, 1.22 * upper)
    axes[0, 1].legend(frameon=False, ncol=2)
    rho_positions = np.arange(len(checkpoints))
    axes[1, 0].scatter(
        rho_positions - 0.09,
        [float(row["spearman_pauli_gain50"]) for row in checkpoints],
        color=colors["pauli"],
        s=24,
        label=labels["pauli"],
    )
    axes[1, 0].scatter(
        rho_positions + 0.09,
        [float(row["spearman_gram_gain50"]) for row in checkpoints],
        color=colors["gram"],
        marker="s",
        s=22,
        label=labels["gram"],
    )
    axes[1, 0].axhline(0.0, color="#888888", linewidth=0.7)
    axes[1, 0].set_ylim(-1.05, 1.12)
    axes[1, 0].set_xlabel("Instance (grouped by size)")
    axes[1, 0].set_ylabel("Candidate-level Spearman $\\rho$")
    axes[1, 0].set_title("(c) Local score vs 50-step gain")
    axes[1, 0].legend(frameon=False, loc="lower left")
    group_start = 0
    for n in completed_sizes:
        count = sum((int(row["n"]) == n for row in checkpoints))
        axes[1, 0].text(
            group_start + (count - 1) / 2,
            1.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
        group_start += count
        if group_start < len(checkpoints):
            axes[1, 0].axvline(group_start - 0.5, color="#bbbbbb", linewidth=0.6)
    grouped_candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        grouped_candidates[str(row["instance"])].append(row)
    rank_x: list[float] = []
    rank_y: list[float] = []
    rank_color: list[int] = []
    for checkpoint in checkpoints:
        rows = grouped_candidates[str(checkpoint["instance"])]
        rank_x.extend(rankdata([float(row["pauli_score"]) for row in rows]))
        rank_y.extend(rankdata([float(row["gram_residual_score"]) for row in rows]))
        rank_color.extend([int(checkpoint["n"])] * len(rows))
    size_colors = {8: "#7b3294", 10: "#00897b", 12: "#c99700"}
    for n, color in size_colors.items():
        indices = [i for i, size in enumerate(rank_color) if size == n]
        axes[1, 1].scatter(
            [rank_x[i] for i in indices],
            [rank_y[i] for i in indices],
            color=color,
            label=f"$n={n}$",
            s=18,
            alpha=0.7,
            edgecolor="none",
        )
    upper = max(rank_x + rank_y) + 0.5
    axes[1, 1].plot(
        [0.5, upper], [0.5, upper], color="#777777", linestyle=":", linewidth=0.9
    )
    axes[1, 1].set_xlim(0.5, upper)
    axes[1, 1].set_ylim(0.5, upper)
    axes[1, 1].set_xlabel("Within-pool candidate-only rank")
    axes[1, 1].set_ylabel("Within-pool incremental rank")
    changed = sum((int(row["pauli_gram_selected_different"]) for row in checkpoints))
    axes[1, 1].set_title(
        f"(d) Ranking changes; selectors differ in {changed}/{len(checkpoints)} pools"
    )
    axes[1, 1].legend(loc="lower right", frameon=False)
    fig.savefig(output_dir / "fig_xxz_recovery_pilot.pdf")
    plt.close(fig)
