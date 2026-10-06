from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
from collections import defaultdict
from scipy.stats import binomtest, rankdata, spearmanr, wilcoxon

SCORES = ("delta_B", "pauli_score", "osr_score")
CHECKPOINTS = (1, 5, 20, 50)
PAIR_DEFINITIONS = (
    ("projection", "random"),
    ("projection", "osr_only"),
    ("projection", "pauli_only"),
    ("pauli_only", "random"),
)
BOOTSTRAP_SEED = 20260724
CLUSTER_BOOTSTRAP_SAMPLES = 20000
PAIRED_BOOTSTRAP_SAMPLES = 50000
RANDOM_DRAWS_PER_INSTANCE = 1000
TIE_TOLERANCE = 1e-10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_by_instance(
    rows: list[dict[str, str]], nonidentity_only: bool = False
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if nonidentity_only and int(row["candidate_index"]) == 0:
            continue
        grouped[row["instance"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["candidate_index"]))
    return dict(sorted(grouped.items()))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return (float("nan"), float("nan"))
    result = spearmanr(x, y)
    return (float(result.statistic), float(result.pvalue))


def within_standardize(values: np.ndarray, use_ranks: bool) -> np.ndarray:
    transformed = rankdata(values, method="average") if use_ranks else values.copy()
    standard_deviation = float(np.std(transformed, ddof=0))
    if standard_deviation <= 1e-15:
        return np.zeros_like(transformed, dtype=float)
    return (transformed - np.mean(transformed)) / standard_deviation


def cluster_bootstrap_association(
    candidate_groups: dict[str, list[dict[str, str]]],
    score_name: str,
    checkpoint: int,
    use_ranks: bool,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    instances = list(candidate_groups)
    cross_products = []
    score_squares = []
    for instance in instances:
        rows = candidate_groups[instance]
        score = np.array([float(row[score_name]) for row in rows])
        improvement = np.array(
            [float(row[f"improvement_after_{checkpoint}"]) for row in rows]
        )
        score_standardized = within_standardize(score, use_ranks)
        improvement_standardized = within_standardize(improvement, use_ranks)
        cross_products.append(
            float(np.dot(score_standardized, improvement_standardized))
        )
        score_squares.append(float(np.dot(score_standardized, score_standardized)))
    cross_products_array = np.asarray(cross_products)
    score_squares_array = np.asarray(score_squares)
    point = float(np.sum(cross_products_array) / np.sum(score_squares_array))
    bootstrap_indices = rng.integers(
        0, len(instances), size=(CLUSTER_BOOTSTRAP_SAMPLES, len(instances))
    )
    bootstrap_cross = cross_products_array[bootstrap_indices].sum(axis=1)
    bootstrap_square = score_squares_array[bootstrap_indices].sum(axis=1)
    bootstrap = np.divide(
        bootstrap_cross,
        bootstrap_square,
        out=np.full_like(bootstrap_cross, np.nan),
        where=bootstrap_square > 0,
    )
    lower, upper = np.quantile(bootstrap[np.isfinite(bootstrap)], (0.025, 0.975))
    return (point, float(lower), float(upper))


def instance_correlations(
    candidate_groups: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for instance, rows in candidate_groups.items():
        metadata = rows[0]
        for score_name in SCORES:
            score = np.array([float(row[score_name]) for row in rows])
            for checkpoint in CHECKPOINTS:
                improvement = np.array(
                    [float(row[f"improvement_after_{checkpoint}"]) for row in rows]
                )
                rho, pvalue = safe_spearman(score, improvement)
                rows_out.append(
                    {
                        "instance": instance,
                        "seed": int(metadata["seed"]),
                        "n": int(metadata["n"]),
                        "t": int(metadata["t"]),
                        "candidate_count": len(rows),
                        "score": score_name,
                        "checkpoint": checkpoint,
                        "spearman_rho": rho,
                        "spearman_pvalue_descriptive_only": pvalue,
                    }
                )
    return rows_out


def paired_method_comparisons(
    method_rows: list[dict[str, str]], rng: np.random.Generator
) -> list[dict[str, object]]:
    by_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in method_rows:
        by_instance[row["instance"]][row["method"]] = row
    output = []
    for left, right in PAIR_DEFINITIONS:
        differences = np.array(
            [
                float(methods[left]["improvement_after_50"])
                - float(methods[right]["improvement_after_50"])
                for methods in by_instance.values()
            ],
            dtype=float,
        )
        bootstrap_indices = rng.integers(
            0, differences.size, size=(PAIRED_BOOTSTRAP_SAMPLES, differences.size)
        )
        bootstrap_means = differences[bootstrap_indices].mean(axis=1)
        ci_lower, ci_upper = np.quantile(bootstrap_means, (0.025, 0.975))
        wins = int(np.sum(differences > TIE_TOLERANCE))
        losses = int(np.sum(differences < -TIE_TOLERANCE))
        ties = int(differences.size - wins - losses)
        nonzero = differences[np.abs(differences) > TIE_TOLERANCE]
        if nonzero.size:
            wilcoxon_result = wilcoxon(
                nonzero,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
            wilcoxon_statistic = float(wilcoxon_result.statistic)
            wilcoxon_pvalue = float(wilcoxon_result.pvalue)
            sign_pvalue = float(
                binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue
            )
        else:
            wilcoxon_statistic = 0.0
            wilcoxon_pvalue = 1.0
            sign_pvalue = 1.0
        output.append(
            {
                "left_method": left,
                "right_method": right,
                "instances": differences.size,
                "mean_paired_difference": float(np.mean(differences)),
                "median_paired_difference": float(np.median(differences)),
                "paired_bootstrap95_lower": float(ci_lower),
                "paired_bootstrap95_upper": float(ci_upper),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "tie_tolerance": TIE_TOLERANCE,
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_two_sided_p_exploratory": wilcoxon_pvalue,
                "sign_test_two_sided_p_exploratory": sign_pvalue,
            }
        )
    return output


def repeated_random_resampling(
    candidate_groups_all: dict[str, list[dict[str, str]]],
    method_rows: list[dict[str, str]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    method_by_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in method_rows:
        method_by_instance[row["instance"]][row["method"]] = row
    output: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for instance, all_rows in candidate_groups_all.items():
        pool = [row for row in all_rows if int(row["candidate_index"]) != 0]
        improvements = np.array(
            [float(row["improvement_after_50"]) for row in pool], dtype=float
        )
        oracle_improvement = float(
            method_by_instance[instance]["oracle"]["improvement_after_50"]
        )
        near_threshold = 0.9 * oracle_improvement
        near_flags = improvements >= near_threshold - 1e-12
        exact_probability = float(np.mean(near_flags))
        sampled_indices = rng.integers(0, len(pool), size=RANDOM_DRAWS_PER_INSTANCE)
        for draw, pool_index in enumerate(sampled_indices):
            candidate = pool[int(pool_index)]
            output.append(
                {
                    "row_type": "random_draw",
                    "instance": instance,
                    "seed": int(candidate["seed"]),
                    "n": int(candidate["n"]),
                    "t": int(candidate["t"]),
                    "draw": draw,
                    "method": "random",
                    "candidate_index": int(candidate["candidate_index"]),
                    "improvement_after_50": float(candidate["improvement_after_50"]),
                    "near_oracle": int(
                        float(candidate["improvement_after_50"])
                        >= near_threshold - 1e-12
                    ),
                    "oracle_improvement_after_50": oracle_improvement,
                    "near_oracle_threshold": near_threshold,
                    "exact_random_near_oracle_probability": exact_probability,
                    "method_percentile_in_candidate_pool": "",
                }
            )
        method_percentiles: dict[str, float] = {}
        for method in ("osr_only", "pauli_only", "projection"):
            method_row = method_by_instance[instance][method]
            method_improvement = float(method_row["improvement_after_50"])
            percentile = 100.0 * float(
                np.mean(improvements <= method_improvement + 1e-12)
            )
            method_percentiles[method] = percentile
            output.append(
                {
                    "row_type": "method_percentile",
                    "instance": instance,
                    "seed": int(method_row["seed"]),
                    "n": int(method_row["n"]),
                    "t": int(method_row["t"]),
                    "draw": "",
                    "method": method,
                    "candidate_index": int(method_row["candidate_index"]),
                    "improvement_after_50": method_improvement,
                    "near_oracle": int(method_improvement >= near_threshold - 1e-12),
                    "oracle_improvement_after_50": oracle_improvement,
                    "near_oracle_threshold": near_threshold,
                    "exact_random_near_oracle_probability": exact_probability,
                    "method_percentile_in_candidate_pool": percentile,
                }
            )
        summaries[instance] = {
            "n": int(pool[0]["n"]),
            "t": int(pool[0]["t"]),
            "seed": int(pool[0]["seed"]),
            "candidate_count": len(pool),
            "random_mean": float(np.mean(improvements)),
            "random_median": float(np.median(improvements)),
            "random_q025": float(np.quantile(improvements, 0.025)),
            "random_q975": float(np.quantile(improvements, 0.975)),
            "random_exact_near_oracle_probability": exact_probability,
            "percentiles": method_percentiles,
        }
    return (output, summaries)


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return (float(center - half_width), float(center + half_width))


def make_figure(
    method_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    instance_rows: list[dict[str, object]],
    random_summaries: dict[str, dict[str, object]],
    trace_path: Path,
    output_dir: Path,
) -> None:
    colors = {
        "identity": "#7f7f7f",
        "random": "#df9417",
        "osr_only": "#2a9d8f",
        "pauli_only": "#5b8ff9",
        "projection": "#c23b73",
        "oracle": "#222222",
    }
    labels = {
        "identity": "identity control",
        "random": "random resampling",
        "osr_only": "OSR-only",
        "pauli_only": "candidate-only",
        "projection": "incremental",
        "oracle": "oracle",
    }
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.25, 6.35), constrained_layout=True)
    representative = sorted({row["instance"] for row in method_rows})[0]
    representative_rows = [
        row for row in method_rows if row["instance"] == representative
    ]
    representative_map = {row["method"]: row for row in representative_rows}
    old_projected = float(representative_rows[0]["old_projected_norm"])
    jump_methods = ("identity", "random", "osr_only", "pauli_only", "projection")
    jump_values = [old_projected] + [
        np.sqrt(float(representative_map[method]["full_projected_score"]))
        for method in jump_methods
    ]
    jump_labels = [
        "old",
        "identity control",
        "random",
        "OSR-only",
        "candidate-only",
        "incremental",
    ]
    jump_colors = ["#222222"] + [colors[method] for method in jump_methods]
    floor = 1e-13
    axes[0, 0].bar(
        np.arange(len(jump_values)),
        np.maximum(jump_values, floor) - floor,
        bottom=floor,
        width=0.68,
        color=jump_colors,
        alpha=0.84,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(floor, 1.0)
    axes[0, 0].set_xticks(np.arange(len(jump_values)))
    axes[0, 0].set_xticklabels(jump_labels, rotation=22, ha="right")
    axes[0, 0].set_ylabel("$\\|\\Pi_{\\mathcal{T}}G\\|$")
    axes[0, 0].set_title("(a) Instantaneous tangent recovery (one instance)")
    with np.load(trace_path) as archive:
        traces = {key: archive[key] for key in archive.files}
    for method in (
        "identity",
        "random",
        "osr_only",
        "pauli_only",
        "projection",
        "oracle",
    ):
        row = representative_map[method]
        candidate_name = row["candidate_name"]
        key = f"{representative}__{candidate_name}"
        axes[0, 1].plot(
            np.arange(len(traces[key])),
            traces[key],
            color=colors[method],
            linewidth=1.8 if method in ("projection", "oracle") else 1.2,
            linestyle="--" if method == "oracle" else "-",
            label="random" if method == "random" else labels[method],
        )
    metadata = representative_rows[0]
    axes[0, 1].set_title(
        f"(b) Descent after same-point intervention\n$n={metadata['n']}, t={metadata['t']}$"
    )
    axes[0, 1].set_xlabel("Optimization step")
    axes[0, 1].set_ylabel("Infidelity")
    axes[0, 1].legend(frameon=False, ncol=2, loc="center right")
    delta5 = [
        row
        for row in instance_rows
        if row["score"] == "delta_B" and int(row["checkpoint"]) == 5
    ]
    for position, n in enumerate((4, 6, 8), start=1):
        values = np.array(
            [float(row["spearman_rho"]) for row in delta5 if int(row["n"]) == n]
        )
        axes[1, 0].scatter(
            position + np.linspace(-0.14, 0.14, len(values)),
            values,
            s=30,
            color=("#2878b5", "#f28e2b", "#4e9f50")[position - 1],
            zorder=3,
        )
        axes[1, 0].plot(
            [position - 0.2, position + 0.2],
            [np.median(values), np.median(values)],
            color="black",
            linewidth=1.5,
            label="median" if position == 1 else None,
        )
    axes[1, 0].set_xticks((1, 2, 3), ("n=4", "n=6", "n=8"))
    axes[1, 0].set_xlim(0.5, 3.5)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("System size")
    axes[1, 0].set_ylabel("Spearman $\\rho$ (5-step loss decrease)")
    axes[1, 0].set_title("(c) Score-descent rank correlation")
    axes[1, 0].grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[1, 0].legend(frameon=False, loc="lower right")
    methods = ("random", "osr_only", "pauli_only", "projection")
    ns = (4, 6, 8)
    width = 0.18
    method_by_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in method_rows:
        method_by_instance[row["instance"]][row["method"]] = row
    for method_index, method in enumerate(methods):
        for n_index, n in enumerate(ns):
            center = n_index + (method_index - 1.5) * width
            instances = [
                instance
                for instance, summary in random_summaries.items()
                if int(summary["n"]) == n
            ]
            if method == "random":
                values = np.array(
                    [
                        float(
                            random_summaries[instance][
                                "random_exact_near_oracle_probability"
                            ]
                        )
                        for instance in instances
                    ]
                )
                bar_value = float(np.mean(values))
                bootstrap_rng = np.random.default_rng(9000 + n)
                sampled = bootstrap_rng.choice(
                    values, size=(10000, len(values)), replace=True
                ).mean(axis=1)
                lower, upper = np.quantile(sampled, (0.025, 0.975))
            else:
                values = np.array(
                    [
                        int(
                            method_by_instance[instance][method]["success_90pct_oracle"]
                        )
                        for instance in instances
                    ],
                    dtype=float,
                )
                successes = int(np.sum(values))
                bar_value = float(np.mean(values))
                lower, upper = wilson_interval(successes, len(values))
                annotation = f"{successes}/{len(values)}"
            axes[1, 1].bar(
                center,
                bar_value,
                width=width * 0.9,
                color=colors[method],
                alpha=0.75,
                label=labels[method] if n_index == 0 else None,
            )
            axes[1, 1].errorbar(
                center,
                bar_value,
                yerr=np.array([[bar_value - lower], [upper - bar_value]]),
                color="#222222",
                capsize=2.2,
                linewidth=0.8,
            )
            point_jitter = np.linspace(-0.045, 0.045, len(values))
            axes[1, 1].scatter(
                center + point_jitter,
                values,
                s=11,
                facecolor="white",
                edgecolor=colors[method],
                linewidth=0.8,
                zorder=3,
            )
            if method != "random":
                axes[1, 1].text(
                    center,
                    min(1.12, upper + 0.055),
                    annotation,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
    axes[1, 1].set_xticks(np.arange(len(ns)), [f"n={n}" for n in ns])
    axes[1, 1].set_ylim(-0.04, 1.22)
    axes[1, 1].set_ylabel("Fraction reaching $\\geq90\\%$ oracle gain")
    axes[1, 1].set_title("(d) Near-oracle success (5 instances per $n$)")
    axes[1, 1].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        columnspacing=0.9,
        handletextpad=0.4,
    )
    (output_dir / "fig_multiblock_stagnation_v2.pdf").parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(output_dir / "fig_multiblock_stagnation_v2.pdf")
    plt.close(fig)
    make_rank_detail_figure(candidate_rows, output_dir)


def make_rank_detail_figure(
    candidate_rows: list[dict[str, str]], output_dir: Path
) -> None:
    """Plot the 20-step candidate ranks."""
    groups = grouped_by_instance(candidate_rows, nonidentity_only=True)
    fig, ax = plt.subplots(figsize=(5.3, 4.0), constrained_layout=True)
    for n, color in zip((4, 6, 8), ("#2878b5", "#f28e2b", "#4e9f50")):
        x, y = ([], [])
        for rows in groups.values():
            if int(rows[0]["n"]) != n:
                continue
            x.extend(
                (rankdata([float(r["delta_B"]) for r in rows]) - 1) / (len(rows) - 1)
            )
            y.extend(
                (rankdata([float(r["improvement_after_20"]) for r in rows]) - 1)
                / (len(rows) - 1)
            )
        ax.scatter(x, y, s=20, alpha=0.45, color=color, label=f"n={n}")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Within-instance rank of $\\Delta_B$")
    ax.set_ylabel("Within-instance rank of 20-step loss decrease")
    ax.set_title("20-step candidate ranks")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(
        (output_dir / "fig_multiblock_stagnation_v2.pdf").with_name(
            "fig_multiblock_rank_detail.pdf"
        )
    )
    plt.close(fig)


def run(data_dir: Path, output_dir: Path) -> None:
    ROOT = output_dir / "statistics"
    CANDIDATE_INPUT = data_dir / "candidates.csv"
    METHOD_INPUT = data_dir / "results.csv"
    TRACE_INPUT = data_dir / "traces.npz"
    INSTANCE_OUTPUT = ROOT / "multiblock_instance_correlations.csv"
    PAIRED_OUTPUT = ROOT / "multiblock_paired_comparisons.csv"
    RANDOM_OUTPUT = ROOT / "multiblock_random_resampling.csv"
    candidate_rows = read_csv(CANDIDATE_INPUT)
    method_rows = read_csv(METHOD_INPUT)
    candidate_groups = grouped_by_instance(candidate_rows, nonidentity_only=True)
    candidate_groups_all = grouped_by_instance(candidate_rows, nonidentity_only=False)
    if len(candidate_groups) != 15 or any(
        (len(rows) != 12 for rows in candidate_groups.values())
    ):
        raise AssertionError(
            "Expected 15 instances with 12 nonidentity candidates each."
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    instance_rows = instance_correlations(candidate_groups)
    write_csv(INSTANCE_OUTPUT, instance_rows)
    association_summary: dict[
        tuple[str, int], dict[str, tuple[float, float, float]]
    ] = {}
    for score_name in SCORES:
        for checkpoint in CHECKPOINTS:
            association_summary[score_name, checkpoint] = {
                "rank": cluster_bootstrap_association(
                    candidate_groups, score_name, checkpoint, use_ranks=True, rng=rng
                ),
                "zscore": cluster_bootstrap_association(
                    candidate_groups, score_name, checkpoint, use_ranks=False, rng=rng
                ),
            }
    paired_rows = paired_method_comparisons(method_rows, rng)
    write_csv(PAIRED_OUTPUT, paired_rows)
    random_rows, random_summaries = repeated_random_resampling(
        candidate_groups_all, method_rows, rng
    )
    write_csv(RANDOM_OUTPUT, random_rows)
    make_figure(
        method_rows,
        candidate_rows,
        instance_rows,
        random_summaries,
        TRACE_INPUT,
        output_dir / "figures",
    )
    summary_json = {
        "instances": len(candidate_groups),
        "candidates_per_instance": 12,
        "cluster_bootstrap_samples": CLUSTER_BOOTSTRAP_SAMPLES,
        "paired_bootstrap_samples": PAIRED_BOOTSTRAP_SAMPLES,
        "random_draws_per_instance": RANDOM_DRAWS_PER_INSTANCE,
        "seed": BOOTSTRAP_SEED,
        "association_summary": {
            f"{score}_{checkpoint}": {
                key: list(value) for key, value in summary.items()
            }
            for (score, checkpoint), summary in association_summary.items()
        },
    }
    (ROOT / "multiblock_hierarchical_manifest.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
