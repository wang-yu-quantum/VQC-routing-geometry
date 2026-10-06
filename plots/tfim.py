from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

CHECKPOINTS = (1, 5, 20, 50)
INTERVENTION_STEPS = 50
OLD_BLOCKS = 2
METHODS = ("identity", "random", "osr_only", "pauli_only", "projection", "oracle")


def score_correlations(
    candidate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        if int(row["candidate_index"]) != 0:
            grouped[str(row["instance"])].append(row)
    outputs: list[dict[str, object]] = []
    score_fields = (
        ("OSR", "osr_score"),
        ("Pauli", "pauli_score"),
        ("projection", "delta_B"),
    )
    for instance, rows in grouped.items():
        for score_name, score_field in score_fields:
            x = np.asarray([float(row[score_field]) for row in rows])
            for checkpoint in CHECKPOINTS:
                y = np.asarray([float(row[f"gain_after_{checkpoint}"]) for row in rows])
                if np.unique(x).size < 2 or np.unique(y).size < 2:
                    rho = float("nan")
                else:
                    rho = float(spearmanr(x, y).statistic)
                outputs.append(
                    {
                        "instance": instance,
                        "n": int(rows[0]["n"]),
                        "score": score_name,
                        "checkpoint": checkpoint,
                        "spearman_rho": rho,
                    }
                )
    return outputs


def make_figure(
    result_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    traces: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.4,
            "axes.labelsize": 9,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.2,
            "figure.dpi": 180,
        }
    )
    colors = {
        "identity": "#777777",
        "random": "#e69f00",
        "osr_only": "#2a9d8f",
        "pauli_only": "#5b8ff9",
        "projection": "#c44e7d",
        "oracle": "#222222",
    }
    labels = {
        "identity": "identity",
        "random": "random",
        "osr_only": "OSR-only",
        "pauli_only": "candidate-only",
        "projection": "incremental",
        "oracle": "oracle",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.25, 6.35), constrained_layout=True)
    method_rows = [row for row in result_rows if row["row_type"] == "method_summary"]
    representative_summary = sorted(
        [summary for summary in summaries if int(summary["n"]) == 8],
        key=lambda item: int(item["seed"]),
    )[0]
    representative = str(representative_summary["instance"])
    representative_methods = {
        str(row["method"]): row
        for row in method_rows
        if row["instance"] == representative
    }
    representative_candidates = [
        row for row in candidate_rows if row["instance"] == representative
    ]
    ground_energy = float(representative_summary["ground_energy"])
    nonidentity_traces = np.stack(
        [
            traces[f"{representative}__{row['candidate_name']}"]
            for row in representative_candidates
            if int(row["candidate_index"]) != 0
        ]
    )
    steps = np.arange(INTERVENTION_STEPS + 1)
    random_error = nonidentity_traces - ground_energy
    axes[0, 0].fill_between(
        steps,
        np.quantile(random_error, 0.1, axis=0),
        np.quantile(random_error, 0.9, axis=0),
        color=colors["random"],
        alpha=0.16,
        label="random 10--90%",
    )
    axes[0, 0].plot(
        steps,
        np.mean(random_error, axis=0),
        color=colors["random"],
        linewidth=1.2,
        label="random mean",
    )
    for method in ("identity", "osr_only", "pauli_only", "projection", "oracle"):
        row = representative_methods[method]
        candidate_name = str(row["candidate_name"])
        values = traces[f"{representative}__{candidate_name}"] - ground_energy
        axes[0, 0].plot(
            steps,
            values,
            color=colors[method],
            linewidth=1.8 if method in ("projection", "oracle") else 1.15,
            linestyle="--" if method == "oracle" else "-",
            label=labels[method],
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Optimization step")
    axes[0, 0].set_ylabel("Energy error $E-E_0$")
    axes[0, 0].set_title(f"(a) Same-point TFIM intervention\nn=8, t={OLD_BLOCKS}")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    nonidentity_rows = [
        row for row in representative_candidates if int(row["candidate_index"]) != 0
    ]
    scatter = axes[0, 1].scatter(
        [max(float(row["pauli_score"]), 1e-16) for row in nonidentity_rows],
        [max(float(row["delta_B"]), 1e-16) for row in nonidentity_rows],
        c=[float(row["overlap_fraction"]) for row in nonidentity_rows],
        cmap="viridis",
        s=31,
        edgecolor="white",
        linewidth=0.5,
    )
    minimum = min(
        min((float(row["pauli_score"]) for row in nonidentity_rows)),
        min((float(row["delta_B"]) for row in nonidentity_rows)),
    )
    maximum = max(
        max((float(row["pauli_score"]) for row in nonidentity_rows)),
        max((float(row["delta_B"]) for row in nonidentity_rows)),
    )
    minimum = max(minimum, 1e-16)
    axes[0, 1].plot(
        [minimum, maximum],
        [minimum, maximum],
        color="#888888",
        linewidth=0.8,
        linestyle=":",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Candidate-only score $\\|\\Pi_{\\mathcal{D}_B}G\\|^2$")
    axes[0, 1].set_ylabel("Incremental gain $\\Delta_B$")
    axes[0, 1].set_title("(b) Candidate-only and incremental alignment")
    colorbar = fig.colorbar(scatter, ax=axes[0, 1], pad=0.01)
    colorbar.set_label("overlap fraction with old tangent")
    plot_methods = ("random", "osr_only", "pauli_only", "projection")
    ns = (4, 6, 8)
    width = 0.18
    for method_index, method in enumerate(plot_methods):
        for n_index, n in enumerate(ns):
            values = np.asarray(
                [
                    float(row["gain_50"])
                    for row in method_rows
                    if row["method"] == method and int(row["n"]) == n
                ]
            )
            center = n_index + (method_index - 1.5) * width
            axes[1, 0].bar(
                center,
                float(np.mean(values)),
                width=width * 0.88,
                color=colors[method],
                alpha=0.7,
                label=labels[method] if n_index == 0 else None,
            )
            jitter = np.linspace(-0.045, 0.045, len(values))
            axes[1, 0].scatter(
                center + jitter,
                values,
                s=13,
                facecolor="white",
                edgecolor=colors[method],
                linewidth=0.8,
                zorder=3,
            )
    axes[1, 0].set_xticks(range(len(ns)), [f"n={n}" for n in ns])
    axes[1, 0].set_ylabel("50-step energy decrease")
    axes[1, 0].set_title("(c) Selection gains after retraining")
    axes[1, 0].legend(
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14)
    )
    correlations = score_correlations(candidate_rows)
    score_order = ("OSR", "Pauli", "projection")
    score_colors = ("#2a9d8f", "#5b8ff9", "#c44e7d")
    positions = np.arange(len(score_order))
    for position, (score, color) in enumerate(zip(score_order, score_colors)):
        values = np.asarray(
            [
                float(row["spearman_rho"])
                for row in correlations
                if row["score"] == score and int(row["checkpoint"]) == 20
            ]
        )
        jitter = np.linspace(-0.12, 0.12, len(values))
        axes[1, 1].scatter(position + jitter, values, s=15, color=color, alpha=0.65)
        axes[1, 1].plot(
            [position - 0.18, position + 0.18],
            [np.median(values), np.median(values)],
            color="#222222",
            linewidth=1.2,
        )
    axes[1, 1].axhline(0.0, color="#888888", linewidth=0.7)
    axes[1, 1].set_xticks(positions, ["OSR-only", "candidate-only", "incremental"])
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set_ylabel("Per-instance Spearman $\\rho$")
    axes[1, 1].set_title(
        f"(d) Score vs 20-step decrease\n{len(summaries)} instances, {len(nonidentity_rows)} candidates each"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_natural_hamiltonian_intervention.pdf")
    plt.close(fig)
