from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from collections import defaultdict
from scipy.stats import spearmanr

BASE_SEED = 20260723
SIZES = (4, 6, 8)
FAMILIES = (
    "line_brickwork",
    "ring",
    "random_fixed_budget",
    "repeated_dimer",
    "matched_cut_budget",
)


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2)) / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def cluster_bootstrap_interval(
    rows: list[dict[str, object]],
    event_key: str,
    rng: np.random.Generator,
    samples: int = 4000,
) -> tuple[float, float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["route_id"])].append(float(row[event_key]))
    keys = sorted(clusters)
    if not keys:
        return (float("nan"), float("nan"))
    draws = np.empty(samples)
    for sample in range(samples):
        selected = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in selected for value in clusters[str(key)]]
        draws[sample] = float(np.mean(values))
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def grouped(
    rows: list[dict[str, object]], **filters: object
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if all((row[key] == value for key, value in filters.items()))
    ]


def finite_spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def make_figure(rows: list[dict[str, object]], *, output_dir: Path | None) -> None:
    labels = {
        "line_brickwork": "line",
        "ring": "ring",
        "random_fixed_budget": "random",
        "repeated_dimer": "dimer",
        "matched_cut_budget": "matched cut\nbudget",
    }
    colors = {
        "line_brickwork": "#2878B5",
        "ring": "#9C4A9C",
        "random_fixed_budget": "#E07A1F",
        "repeated_dimer": "#3A9D75",
        "matched_cut_budget": "#C84B5A",
    }
    markers = {4: "o", 6: "s", 8: "^"}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    ax = axes[0, 0]
    positions = np.arange(len(FAMILIES))
    data = [
        np.log10(
            np.maximum(
                [
                    float(row["chi_ratio"])
                    for row in grouped(rows, routing_family=family)
                ],
                1e-16,
            )
        )
        for family in FAMILIES
    ]
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#eeeeee", "edgecolor": "#aaaaaa"},
        medianprops={"color": "#888888", "linewidth": 1},
        whiskerprops={"color": "#aaaaaa"},
        capprops={"color": "#aaaaaa"},
    )
    for patch, family in zip(box["boxes"], FAMILIES):
        patch.set_facecolor(colors[family])
        patch.set_alpha(0.2)
    size_offsets = {4: -0.18, 6: 0.0, 8: 0.18}
    rng = np.random.default_rng(BASE_SEED + 41)
    for index, family in enumerate(FAMILIES):
        subset = grouped(rows, routing_family=family)
        for n in SIZES:
            values = [
                math.log10(max(float(row["chi_ratio"]), 1e-16))
                for row in subset
                if int(row["n"]) == n
            ]
            jitter = rng.uniform(-0.025, 0.025, len(values))
            ax.scatter(
                index + size_offsets[n] + jitter,
                values,
                s=16,
                marker=markers[n],
                color=colors[family],
                alpha=0.72,
                edgecolor="none",
                zorder=3,
            )
    for n in SIZES:
        ax.scatter(
            [],
            [],
            s=22,
            marker=markers[n],
            facecolors="none",
            edgecolors="#555555",
            linewidths=0.9,
            label=f"n={n}",
        )
    ax.axhline(
        -3, color="black", linestyle="--", linewidth=1, label="primary threshold"
    )
    ax.set_xticks(positions, [labels[family] for family in FAMILIES])
    ax.set_ylabel("$\\log_{10}\\chi_{\\rm ratio}$")
    ax.set_title("(a) Full accessible-gradient distribution")
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.56, 0.98),
        handletextpad=0.5,
    )
    ax = axes[0, 1]
    offsets = {4: -0.18, 6: 0.0, 8: 0.18}
    for n in SIZES:
        for position, family in zip(positions, FAMILIES):
            subset = grouped(rows, n=n, routing_family=family)
            successes = sum((int(row["event_primary"]) for row in subset))
            lo, hi = wilson_interval(successes, len(subset))
            frequency = successes / len(subset)
            ax.errorbar(
                position + offsets[n],
                frequency,
                yerr=np.array([[max(0.0, frequency - lo)], [max(0.0, hi - frequency)]]),
                fmt=markers[n],
                color=colors[family],
                capsize=3,
                linewidth=1.2,
            )
        ax.scatter(
            [],
            [],
            s=22,
            marker=markers[n],
            facecolors="none",
            edgecolors="#555555",
            linewidths=0.9,
            label=f"n={n}",
        )
    ax.set_xticks(positions, [labels[family] for family in FAMILIES])
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("primary near-stagnation frequency")
    ax.set_title("(b) Empirical frequency with Wilson intervals")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.18, 0.99))
    ax = axes[1, 0]
    for family in FAMILIES:
        subset = grouped(rows, routing_family=family)
        for n in SIZES:
            selected = [row for row in subset if int(row["n"]) == n]
            ax.scatter(
                [float(row["energy_error"]) for row in selected],
                [max(float(row["chi_ratio"]), 1e-16) for row in selected],
                color=colors[family],
                marker=markers[n],
                s=22,
                alpha=0.7,
                edgecolor="none",
            )
    ax.axhline(0.001, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.01, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_limits, y_limits = (ax.get_xlim(), ax.get_ylim())
    ax.fill_between(
        [0.01, x_limits[1]], y_limits[0], 0.001, color="#DFEBE7", alpha=0.55, zorder=0
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.text(
        0.3,
        0.14,
        "Finite error,\nlow coverage",
        transform=ax.transAxes,
        fontsize=9,
        color="#405A50",
    )
    ax.set_xlabel("final energy error $E-E_0$")
    ax.set_ylabel("$\\chi_{\\rm ratio}$")
    ax.set_title("(c) Low coverage away from the ground state")
    ax = axes[1, 1]
    for family in FAMILIES:
        subset = grouped(rows, routing_family=family)
        for n in SIZES:
            selected = [row for row in subset if int(row["n"]) == n]
            ax.scatter(
                [float(row["total_cut_budget"]) for row in selected],
                [max(float(row["chi_ratio"]), 1e-16) for row in selected],
                color=colors[family],
                marker=markers[n],
                s=22,
                alpha=0.68,
                edgecolor="none",
            )
        ax.plot([], [], color=colors[family], linewidth=3, label=labels[family])
    ax.set_yscale("log")
    ax.set_xlabel("scalar cumulative contiguous-cut budget")
    ax.set_ylabel("$\\chi_{\\rm ratio}$")
    ax.set_title("(d) Same scalar cut budget, different coverage")
    matched = next(
        (
            row
            for row in rows
            if row["route_id"] == "n6_r18_matched_cut_budget_v2"
            and int(row["initialization_index"]) == 2
        )
    )
    reference = next(
        (
            row
            for row in rows
            if row["route_id"] == matched["paired_route_id"]
            and int(row["initialization_index"]) == 2
        )
    )
    budget = float(matched["total_cut_budget"])
    low, high = sorted((float(row["chi_ratio"]) for row in (matched, reference)))
    ax.scatter(
        [budget, budget],
        [low, high],
        s=65,
        facecolors="none",
        marker=markers[int(matched["n"])],
        edgecolors="#444444",
        linewidths=0.9,
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(budget + 0.65, high),
        xytext=(budget + 0.65, low),
        arrowprops={"arrowstyle": "|-|", "color": "#555555", "lw": 0.9},
    )
    ax.text(
        budget + 1.2,
        0.006,
        "Paired random–matched\nexample: $n=6$, budget = 11",
        fontsize=8,
        color="#444444",
        va="center",
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="center right",
        bbox_to_anchor=(0.99, 0.48),
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "fig_stagnation_genericity_distribution.pdf"
        if output_dir is None
        else output_dir
        / (output_dir / "fig_stagnation_genericity_distribution.pdf").name,
        bbox_inches="tight",
    )
    plt.close(fig)


def read_results(path: Path) -> list[dict[str, object]]:
    integer_fields = {
        "n",
        "route_index",
        "variant",
        "initialization_index",
        "initialization_seed",
        "parameter_count",
        "routing_blocks",
        "cnot_slots_per_block",
        "total_cnot_count",
        "total_cut_budget",
        "tangent_rank",
        "event_relaxed",
        "event_primary",
        "event_strict",
        "optimizer_status",
        "optimizer_iterations",
        "optimizer_evaluations",
    }
    boolean_fields = {"optimizer_success"}
    string_fields = {
        "route_id",
        "routing_family",
        "paired_route_id",
        "routing_sequences",
        "cumulative_cut_profile",
        "optimizer_message",
    }
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in raw.items():
                if key in integer_fields:
                    row[key] = int(value)
                elif key in boolean_fields:
                    row[key] = value == "True"
                elif key in string_fields:
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def statistics(rows: list[dict[str, object]]) -> dict[str, object]:
    rng = np.random.default_rng(BASE_SEED + 909)
    frequency_rows: list[list[object]] = []
    for n in SIZES:
        for family in FAMILIES:
            subset = grouped(rows, n=n, routing_family=family)
            successes = sum((int(row["event_primary"]) for row in subset))
            wilson = wilson_interval(successes, len(subset))
            cluster = cluster_bootstrap_interval(subset, "event_primary", rng)
            route_successes = len(
                {
                    str(row["route_id"])
                    for row in subset
                    if int(row["event_primary"]) == 1
                }
            )
            frequency_rows.append(
                [
                    n,
                    family,
                    f"{successes}/{len(subset)}",
                    f"[{wilson[0]:.3f}, {wilson[1]:.3f}]",
                    f"[{cluster[0]:.3f}, {cluster[1]:.3f}]",
                    f"{route_successes}/4",
                ]
            )
    overall_rows: list[list[object]] = []
    for family in FAMILIES:
        subset = grouped(rows, routing_family=family)
        values = np.array([float(row["chi_ratio"]) for row in subset])
        successes = sum((int(row["event_primary"]) for row in subset))
        cluster = cluster_bootstrap_interval(subset, "event_primary", rng)
        overall_rows.append(
            [
                family,
                f"{successes}/{len(subset)}",
                f"[{cluster[0]:.3f}, {cluster[1]:.3f}]",
                f"{np.median(values):.3e}",
                f"[{np.quantile(values, 0.25):.3e}, {np.quantile(values, 0.75):.3e}]",
            ]
        )
    sensitivity_rows = []
    for event_name in ("relaxed", "primary", "strict"):
        key = f"event_{event_name}"
        successes = sum((int(row[key]) for row in rows))
        route_successes = len(
            {str(row["route_id"]) for row in rows if int(row[key]) == 1}
        )
        sensitivity_rows.append(
            [event_name, f"{successes}/{len(rows)}", f"{route_successes}/60"]
        )
    relation_rows = []
    for n in SIZES:
        subset = grouped(rows, n=n)
        log_chi = [math.log10(max(float(row["chi_ratio"]), 1e-16)) for row in subset]
        relation_rows.append(
            [
                n,
                f"{finite_spearman([float(row['total_cut_budget']) for row in subset], log_chi):.3f}",
                f"{finite_spearman([float(row['tangent_rank']) for row in subset], log_chi):.3f}",
                f"{finite_spearman([float(row['energy_error']) for row in subset], log_chi):.3f}",
            ]
        )
    primary_rows = [row for row in rows if int(row["event_primary"]) == 1]
    primary_routes = sorted({str(row["route_id"]) for row in primary_rows})
    ill_conditioned = [
        row
        for row in primary_rows
        if float(row["gram_condition_number"]) > 100000000000000.0
    ]
    clean_primary = [
        row
        for row in primary_rows
        if bool(row["optimizer_success"])
        and float(row["gram_condition_number"]) <= 100000000000000.0
    ]
    clean_primary_routes = sorted({str(row["route_id"]) for row in clean_primary})
    convergence_failures = [row for row in rows if not bool(row["optimizer_success"])]
    criterion_passes = {
        "parameter gradient": sum(
            (float(row["parameter_gradient_norm"]) <= 1e-06 for row in rows)
        ),
        "ambient gradient": sum(
            (float(row["ambient_gradient_norm"]) >= 0.01 for row in rows)
        ),
        "chi ratio": sum((float(row["chi_ratio"]) <= 0.001 for row in rows)),
        "energy error": sum((float(row["energy_error"]) >= 0.01 for row in rows)),
    }
    gradient_mismatch = max(
        (float(row["analytic_vs_geometry_gradient_error"]) for row in rows)
    )
    median_seconds = float(np.median([float(row["elapsed_seconds"]) for row in rows]))
    total_seconds = float(sum((float(row["elapsed_seconds"]) for row in rows)))
    stable_signal = bool(len(clean_primary) > 0 and len(clean_primary_routes) >= 2)
    return {
        "by_size": frequency_rows,
        "by_family": overall_rows,
        "sensitivity": sensitivity_rows,
        "rank_correlations": relation_rows,
        "primary_instances": len(primary_rows),
        "primary_routes": len(primary_routes),
        "ill_conditioned_primary": len(ill_conditioned),
        "well_conditioned_primary": len(clean_primary),
        "well_conditioned_primary_routes": len(clean_primary_routes),
        "optimizer_failures": len(convergence_failures),
        "criterion_passes": criterion_passes,
        "max_gradient_mismatch": gradient_mismatch,
        "original_run_median_seconds": median_seconds,
        "original_run_total_seconds": total_seconds,
        "stable_signal": stable_signal,
    }
