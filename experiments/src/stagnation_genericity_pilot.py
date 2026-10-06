from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import spearmanr

from multiblock_stagnation_intervention import Gate, add_cnot_sequence, add_local_layer
from natural_hamiltonian_intervention import (
    adam_optimize,
    energy_and_gradient,
    energy_geometry,
    exact_ground_state,
    initial_product_theta,
    tfim_hamiltonian,
)
from VQC.routing_geometry.core import kappa, routing_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "stagnation_genericity_manifest.json"
RESULTS_PATH = ROOT / "stagnation_genericity_results.csv"
REPORT_PATH = ROOT / 'stagnation_genericity_pilot_statistics.json'
FIGURE_PDF = ROOT / "fig_stagnation_genericity_distribution.pdf"
FIGURE_PNG = ROOT / "fig_stagnation_genericity_distribution.png"

BASE_SEED = 20260723
SIZES = (4, 6, 8)
FAMILIES = (
    "line_brickwork",
    "ring",
    "random_fixed_budget",
    "repeated_dimer",
    "matched_cut_budget",
)
ROUTES_PER_FAMILY = 4
INITIALIZATIONS = 3
RANK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class RouteSpec:
    n: int
    route_index: int
    family: str
    variant: int
    blocks: tuple[tuple[tuple[int, int], ...], ...]
    matrices: tuple[np.ndarray, ...]
    cumulative_cut_profile: tuple[int, ...]
    total_cut_budget: int
    paired_route_id: str

    @property
    def route_id(self) -> str:
        return f"n{self.n}_r{self.route_index:02d}_{self.family}_v{self.variant}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = manifest["events"]["primary"]
    if expected != {
        "parameter_gradient_max": 1e-6,
        "ambient_gradient_min": 1e-2,
        "chi_ratio_max": 1e-3,
        "energy_error_min": 1e-2,
    }:
        raise RuntimeError("Primary thresholds no longer match the preregistration.")
    return manifest


def locked_protocol_hash(manifest: dict[str, object]) -> str:
    locked = {
        key: value
        for key, value in manifest.items()
        if key not in {"status", "execution"}
    }
    payload = json.dumps(locked, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_manifest(manifest: dict[str, object]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def directed(edges: list[tuple[int, int]], reverse: bool) -> tuple[tuple[int, int], ...]:
    if reverse:
        return tuple((target, control) for control, target in edges)
    return tuple(edges)


def line_matching(n: int, offset: int, reverse: bool) -> tuple[tuple[int, int], ...]:
    budget = n // 2
    positions: list[int] = []
    candidate = offset % (n - 1)
    while len(positions) < budget:
        if candidate not in positions:
            positions.append(candidate)
        candidate = (candidate + 2) % (n - 1)
    return directed([(q, q + 1) for q in positions], reverse)


def ring_matching(n: int, offset: int, reverse: bool) -> tuple[tuple[int, int], ...]:
    positions = [int((offset + 2 * step) % n) for step in range(n // 2)]
    return directed([(q, (q + 1) % n) for q in positions], reverse)


def random_sequence(n: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    sequence = []
    for _ in range(n // 2):
        control, target = rng.choice(n, size=2, replace=False)
        sequence.append((int(control), int(target)))
    return tuple(sequence)


def route_matrices(
    n: int,
    blocks: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[np.ndarray, ...]:
    return tuple(routing_matrix(n, sequence) for sequence in blocks)


def cut_profile(matrices: tuple[np.ndarray, ...], n: int) -> tuple[int, ...]:
    return tuple(
        sum(kappa(matrix, tuple(range(boundary))) for matrix in matrices)
        for boundary in range(1, n)
    )


def make_route(
    n: int,
    route_index: int,
    family: str,
    variant: int,
    blocks: tuple[tuple[tuple[int, int], ...], ...],
    paired_route_id: str = "",
) -> RouteSpec:
    matrices = route_matrices(n, blocks)
    profile = cut_profile(matrices, n)
    return RouteSpec(
        n=n,
        route_index=route_index,
        family=family,
        variant=variant,
        blocks=blocks,
        matrices=matrices,
        cumulative_cut_profile=profile,
        total_cut_budget=int(sum(profile)),
        paired_route_id=paired_route_id,
    )


def route_key(route: RouteSpec) -> tuple[bytes, ...]:
    return tuple(matrix.tobytes() for matrix in route.matrices)


def build_routes(n: int) -> list[RouteSpec]:
    rng = np.random.default_rng(BASE_SEED + 7919 * n)
    routes: list[RouteSpec] = []
    seen: set[tuple[bytes, ...]] = set()

    def append_unique(route: RouteSpec) -> None:
        key = route_key(route)
        if key in seen:
            raise RuntimeError(f"Duplicate routing tuple generated: {route.route_id}")
        seen.add(key)
        routes.append(route)

    for variant in range(ROUTES_PER_FAMILY):
        reverse = variant >= 2
        blocks = (
            line_matching(n, variant, reverse),
            line_matching(n, variant + 1, not reverse),
        )
        append_unique(make_route(n, len(routes), FAMILIES[0], variant, blocks))

    for variant in range(ROUTES_PER_FAMILY):
        reverse = variant >= 2
        blocks = (
            ring_matching(n, variant, reverse),
            ring_matching(n, variant + 1, not reverse),
        )
        append_unique(make_route(n, len(routes), FAMILIES[1], variant, blocks))

    random_routes: list[RouteSpec] = []
    attempts = 0
    while len(random_routes) < ROUTES_PER_FAMILY and attempts < 10000:
        attempts += 1
        blocks = (random_sequence(n, rng), random_sequence(n, rng))
        route = make_route(n, len(routes), FAMILIES[2], len(random_routes), blocks)
        if route_key(route) in seen:
            continue
        append_unique(route)
        random_routes.append(route)

    for variant in range(ROUTES_PER_FAMILY):
        reverse = variant >= 2
        matching = ring_matching(n, variant % 2, reverse)
        blocks = (matching, matching)
        append_unique(make_route(n, len(routes), FAMILIES[3], variant, blocks))

    for variant, reference in enumerate(random_routes):
        attempts = 0
        while attempts < 200000:
            attempts += 1
            blocks = (random_sequence(n, rng), random_sequence(n, rng))
            candidate = make_route(
                n,
                len(routes),
                FAMILIES[4],
                variant,
                blocks,
                paired_route_id=reference.route_id,
            )
            if (
                candidate.total_cut_budget == reference.total_cut_budget
                and route_key(candidate) not in seen
            ):
                append_unique(candidate)
                break
        else:
            raise RuntimeError(
                f"Could not match cut budget {reference.total_cut_budget} for n={n}."
            )

    if len(routes) != len(FAMILIES) * ROUTES_PER_FAMILY:
        raise AssertionError("Unexpected routing count.")
    return routes


def build_ansatz(route: RouteSpec) -> tuple[list[Gate], int]:
    ops: list[Gate] = []
    parameter = add_local_layer(ops, route.n, 0)
    for sequence in route.blocks:
        add_cnot_sequence(ops, sequence)
        parameter = add_local_layer(ops, route.n, parameter)
    return ops, parameter


def optimize(
    ops: list[Gate],
    initial_theta: np.ndarray,
    hamiltonian,
    n: int,
    manifest: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    config = manifest["optimizer"]
    warmup = adam_optimize(
        ops,
        initial_theta,
        hamiltonian,
        n,
        steps=int(config["adam_steps"]),
        learning_rate=float(config["adam_learning_rate"]),
        cosine_decay=bool(config["adam_cosine_decay"]),
    )

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        energy, gradient, _ = energy_and_gradient(ops, theta, hamiltonian, n)
        return energy, gradient

    result = minimize(
        objective,
        warmup.theta,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(config["lbfgsb_maxiter"]),
            "gtol": float(config["gtol"]),
            "ftol": float(config["ftol"]),
            "maxls": int(config["maxls"]),
        },
    )
    return np.asarray(result.x, dtype=float), {
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "optimizer_evaluations": int(result.nfev),
        "warmup_final_energy": float(warmup.energies[-1]),
    }


def event_indicator(
    gradient_norm: float,
    ambient_norm: float,
    chi_ratio: float,
    energy_error: float,
    thresholds: dict[str, float],
) -> bool:
    return bool(
        gradient_norm <= float(thresholds["parameter_gradient_max"])
        and ambient_norm >= float(thresholds["ambient_gradient_min"])
        and chi_ratio <= float(thresholds["chi_ratio_max"])
        and energy_error >= float(thresholds["energy_error_min"])
    )


def analyze_instance(
    route: RouteSpec,
    initialization_index: int,
    initialization_seed: int,
    hamiltonian,
    ground_energy: float,
    manifest: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(initialization_seed)
    ops, parameter_count = build_ansatz(route)
    initial_theta = initial_product_theta(route.n, 2, rng)
    if initial_theta.size != parameter_count:
        raise AssertionError("Parameter count mismatch.")
    started = time.perf_counter()
    theta, optimizer = optimize(ops, initial_theta, hamiltonian, route.n, manifest)
    elapsed = time.perf_counter() - started
    energy, gradient, _ = energy_and_gradient(ops, theta, hamiltonian, route.n)
    geometry = energy_geometry(ops, theta, hamiltonian, route.n)
    tangent_singular = np.linalg.svd(geometry.tangent_real, compute_uv=False)
    if tangent_singular.size and tangent_singular[0] > 0:
        retained = tangent_singular[
            tangent_singular > RANK_TOLERANCE * tangent_singular[0]
        ]
    else:
        retained = np.empty(0)
    tangent_rank = int(retained.size)
    smallest_tangent = float(retained[-1]) if retained.size else 0.0
    smallest_gram = smallest_tangent**2
    largest_gram = float(retained[0] ** 2) if retained.size else 0.0
    gram_condition = (
        largest_gram / smallest_gram if smallest_gram > 0 else float("inf")
    )
    gradient_norm = float(np.linalg.norm(gradient))
    projected_norm = float(geometry.projected_norm)
    ambient_norm = float(geometry.ambient_norm)
    chi_ratio = projected_norm / ambient_norm if ambient_norm > 1e-15 else 0.0
    energy_error = float(max(0.0, energy - ground_energy))
    events = manifest["events"]
    row: dict[str, object] = {
        "n": route.n,
        "route_id": route.route_id,
        "route_index": route.route_index,
        "routing_family": route.family,
        "variant": route.variant,
        "paired_route_id": route.paired_route_id,
        "initialization_index": initialization_index,
        "initialization_seed": initialization_seed,
        "parameter_count": parameter_count,
        "routing_blocks": len(route.blocks),
        "cnot_slots_per_block": route.n // 2,
        "total_cnot_count": route.n,
        "routing_sequences": json.dumps(route.blocks, separators=(",", ":")),
        "cumulative_cut_profile": json.dumps(route.cumulative_cut_profile),
        "total_cut_budget": route.total_cut_budget,
        "energy": float(energy),
        "ground_energy": float(ground_energy),
        "energy_error": energy_error,
        "parameter_gradient_norm": gradient_norm,
        "ambient_gradient_norm": ambient_norm,
        "projected_gradient_norm": projected_norm,
        "chi_ratio": chi_ratio,
        "chi_squared": chi_ratio**2,
        "tangent_rank": tangent_rank,
        "smallest_retained_tangent_singular_value": smallest_tangent,
        "smallest_retained_gram_eigenvalue": smallest_gram,
        "gram_condition_number": gram_condition,
        "analytic_vs_geometry_gradient_error": float(
            np.linalg.norm(gradient - geometry.parameter_gradient)
        ),
        "event_relaxed": int(
            event_indicator(
                gradient_norm, ambient_norm, chi_ratio, energy_error, events["relaxed"]
            )
        ),
        "event_primary": int(
            event_indicator(
                gradient_norm, ambient_norm, chi_ratio, energy_error, events["primary"]
            )
        ),
        "event_strict": int(
            event_indicator(
                gradient_norm, ambient_norm, chi_ratio, energy_error, events["strict"]
            )
        ),
        "elapsed_seconds": elapsed,
    }
    row.update(optimizer)
    return row


def write_csv(rows: list[dict[str, object]]) -> None:
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


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
        return float("nan"), float("nan")
    draws = np.empty(samples)
    for sample in range(samples):
        selected = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in selected for value in clusters[str(key)]]
        draws[sample] = float(np.mean(values))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def grouped(rows: list[dict[str, object]], **filters: object) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if all(row[key] == value for key, value in filters.items())
    ]


def finite_spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def make_figure(rows: list[dict[str, object]], *, output_dir: Path | None = None) -> None:
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
        np.log10(np.maximum([float(row["chi_ratio"]) for row in grouped(rows, routing_family=family)], 1e-16))
        for family in FAMILIES
    ]
    box = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#eeeeee", "edgecolor": "#aaaaaa"},
        medianprops={"color": "#888888", "linewidth": 1},
        whiskerprops={"color": "#aaaaaa"},
        capprops={"color": "#aaaaaa"},
    )
    for patch, family in zip(box["boxes"], FAMILIES):
        patch.set_facecolor(colors[family])
        patch.set_alpha(0.20)
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
        ax.scatter([], [], s=22, marker=markers[n], facecolors="none",
                   edgecolors="#555555", linewidths=0.9, label=f"n={n}")
    ax.axhline(-3, color="black", linestyle="--", linewidth=1, label="primary threshold")
    ax.set_xticks(positions, [labels[family] for family in FAMILIES])
    ax.set_ylabel(r"$\log_{10}\chi_{\rm ratio}$")
    ax.set_title("(a) Full accessible-gradient distribution")
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper left",
              bbox_to_anchor=(0.56, 0.98), handletextpad=0.5)

    ax = axes[0, 1]
    offsets = {4: -0.18, 6: 0.0, 8: 0.18}
    for n in SIZES:
        for position, family in zip(positions, FAMILIES):
            subset = grouped(rows, n=n, routing_family=family)
            successes = sum(int(row["event_primary"]) for row in subset)
            lo, hi = wilson_interval(successes, len(subset))
            frequency = successes / len(subset)
            ax.errorbar(
                position + offsets[n], frequency,
                yerr=np.array([[max(0.0, frequency - lo)],
                               [max(0.0, hi - frequency)]]),
                fmt=markers[n], color=colors[family],
                capsize=3, linewidth=1.2,
            )
        ax.scatter([], [], s=22, marker=markers[n], facecolors="none",
                   edgecolors="#555555", linewidths=0.9,
                   label=f"n={n}")
    ax.set_xticks(positions, [labels[family] for family in FAMILIES])
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("primary near-stagnation frequency")
    ax.set_title("(b) Empirical frequency with Wilson intervals")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.18, 0.99))

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
    ax.axhline(1e-3, color="black", linestyle="--", linewidth=1)
    ax.axvline(1e-2, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
    ax.fill_between([1e-2, x_limits[1]], y_limits[0], 1e-3,
                    color="#DFEBE7", alpha=0.55, zorder=0)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.text(0.30, 0.14, "Finite error,\nlow coverage",
            transform=ax.transAxes, fontsize=9, color="#405A50")
    ax.set_xlabel(r"final energy error $E-E_0$")
    ax.set_ylabel(r"$\chi_{\rm ratio}$")
    ax.set_title("(c) Low coverage away from the ground state")

    ax = axes[1, 1]
    for family in FAMILIES:
        subset = grouped(rows, routing_family=family)
        for n in SIZES:
            selected = [row for row in subset if int(row["n"]) == n]
            ax.scatter(
                [float(row["total_cut_budget"]) for row in selected],
                [max(float(row["chi_ratio"]), 1e-16) for row in selected],
                color=colors[family], marker=markers[n],
                s=22, alpha=0.68, edgecolor="none",
            )
        ax.plot([], [], color=colors[family], linewidth=3,
                label=labels[family])
    ax.set_yscale("log")
    ax.set_xlabel("scalar cumulative contiguous-cut budget")
    ax.set_ylabel(r"$\chi_{\rm ratio}$")
    ax.set_title("(d) Same scalar cut budget, different coverage")
    # Highlight one saved random--matched pair at the same initialization.
    matched = next(row for row in rows
                   if row["route_id"] == "n6_r18_matched_cut_budget_v2"
                   and int(row["initialization_index"]) == 2)
    reference = next(row for row in rows
                     if row["route_id"] == matched["paired_route_id"]
                     and int(row["initialization_index"]) == 2)
    budget = float(matched["total_cut_budget"])
    low, high = sorted(float(row["chi_ratio"]) for row in (matched, reference))
    ax.scatter([budget, budget], [low, high], s=65, facecolors="none",
               marker=markers[int(matched["n"])],
               edgecolors="#444444", linewidths=0.9, zorder=4)
    ax.annotate("", xy=(budget + 0.65, high), xytext=(budget + 0.65, low),
                arrowprops={"arrowstyle": "|-|", "color": "#555555", "lw": 0.9})
    ax.text(budget + 1.2, 0.006, "Paired random\u2013matched\nexample: $n=6$, budget = 11",
            fontsize=8, color="#444444", va="center")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="center right",
              bbox_to_anchor=(0.99, 0.48))

    fig.tight_layout()
    fig.savefig(FIGURE_PDF if output_dir is None else output_dir / FIGURE_PDF.name,
                bbox_inches="tight")
    fig.savefig(FIGURE_PNG if output_dir is None else output_dir / FIGURE_PNG.name,
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], manifest: dict[str, object],
                 *, output_dir: Path | None = None) -> None:
    rng = np.random.default_rng(BASE_SEED + 909)
    frequency_rows: list[list[object]] = []
    for n in SIZES:
        for family in FAMILIES:
            subset = grouped(rows, n=n, routing_family=family)
            successes = sum(int(row["event_primary"]) for row in subset)
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
        successes = sum(int(row["event_primary"]) for row in subset)
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
        successes = sum(int(row[key]) for row in rows)
        route_successes = len(
            {str(row["route_id"]) for row in rows if int(row[key]) == 1}
        )
        sensitivity_rows.append(
            [event_name, f"{successes}/{len(rows)}", f"{route_successes}/60"]
        )

    relation_rows = []
    for n in SIZES:
        subset = grouped(rows, n=n)
        log_chi = [
            math.log10(max(float(row["chi_ratio"]), 1e-16)) for row in subset
        ]
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
        if float(row["gram_condition_number"]) > 1e14
    ]
    clean_primary = [
        row
        for row in primary_rows
        if bool(row["optimizer_success"])
        and float(row["gram_condition_number"]) <= 1e14
    ]
    clean_primary_routes = sorted({str(row["route_id"]) for row in clean_primary})
    convergence_failures = [
        row for row in rows if not bool(row["optimizer_success"])
    ]
    criterion_passes = {
        "parameter gradient": sum(
            float(row["parameter_gradient_norm"]) <= 1e-6 for row in rows
        ),
        "ambient gradient": sum(
            float(row["ambient_gradient_norm"]) >= 1e-2 for row in rows
        ),
        "chi ratio": sum(float(row["chi_ratio"]) <= 1e-3 for row in rows),
        "energy error": sum(float(row["energy_error"]) >= 1e-2 for row in rows),
    }
    gradient_mismatch = max(
        float(row["analytic_vs_geometry_gradient_error"]) for row in rows
    )
    median_seconds = float(np.median([float(row["elapsed_seconds"]) for row in rows]))
    total_seconds = float(sum(float(row["elapsed_seconds"]) for row in rows))
    stable_signal = bool(
        len(clean_primary) > 0
        and len(clean_primary_routes) >= 2
    )

    report = json.dumps({
        "instances": len(rows),
        "frequency_columns": ["n", "family", "events", "wilson95", "cluster95", "routes"],
        "by_size": frequency_rows,
        "family_columns": ["family", "events", "cluster95", "median_coverage", "coverage_iqr"],
        "by_family": overall_rows,
        "sensitivity_columns": ["definition", "events", "routes"],
        "sensitivity": sensitivity_rows,
        "correlation_columns": ["n", "budget_vs_coverage", "rank_vs_coverage", "error_vs_coverage"],
        "rank_correlations": relation_rows,
        "primary_instances": len(primary_rows),
        "primary_routes": len(primary_routes),
        "ill_conditioned_primary": len(ill_conditioned),
        "well_conditioned_primary": len(clean_primary),
        "well_conditioned_primary_routes": len(clean_primary_routes),
        "optimizer_failures": len(convergence_failures),
        "criterion_passes": criterion_passes,
        "maximum_gradient_mismatch": gradient_mismatch,
        "median_instance_seconds": median_seconds,
        "total_instance_seconds": total_seconds,
        "stable_signal": stable_signal,
    }, indent=2) + "\n"
    report_path = REPORT_PATH if output_dir is None else output_dir / REPORT_PATH.name
    report_path.write_text(report, encoding="utf-8")


def read_results() -> list[dict[str, object]]:
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
    with RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
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


def summarize_saved_results(manifest: dict[str, object], output_dir: Path | None = None) -> None:
    output_dir = (output_dir or ROOT / "rerun" / "stagnation_genericity").resolve()
    names = (FIGURE_PDF.name, FIGURE_PNG.name, REPORT_PATH.name,
             "stagnation_genericity_summary.json")
    protected = {path.resolve() for path in
                 (MANIFEST_PATH, RESULTS_PATH, FIGURE_PDF, FIGURE_PNG, REPORT_PATH)}
    if output_dir == ROOT.resolve() or any(
        (output_dir / name).resolve() in protected for name in names
    ):
        raise ValueError("Summary output must not overwrite released files; choose a separate directory.")
    rows = read_results()
    if len(rows) != 180:
        raise RuntimeError(f"Expected 180 saved rows, found {len(rows)}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(rows, output_dir=output_dir)
    write_report(rows, manifest, output_dir=output_dir)
    summary = {
        "mode": "summarize-only",
        "completed_at": utc_now(),
        "script_sha256": sha256(Path(__file__)),
        "source_manifest_sha256": sha256(MANIFEST_PATH),
        "source_results_sha256": sha256(RESULTS_PATH),
        "protocol_sha256": locked_protocol_hash(manifest),
        "instances": len(rows),
        "event_counts": {level: sum(int(row[f"event_{level}"]) for row in rows)
                         for level in ("relaxed", "primary", "strict")},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                     "platform": platform.platform()},
        "outputs_sha256": {name: sha256(output_dir / name) for name in names[:-1]},
    }
    (output_dir / names[-1]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary outputs: {output_dir}")


def run_estimate(manifest: dict[str, object]) -> None:
    estimates: dict[str, float] = {}
    for n in SIZES:
        route = build_routes(n)[0]
        hamiltonian = tfim_hamiltonian(n)
        ground_energy, _ = exact_ground_state(hamiltonian)
        seed = BASE_SEED + 1000 * n
        row = analyze_instance(route, 0, seed, hamiltonian, ground_energy, manifest)
        estimates[str(n)] = float(row["elapsed_seconds"])
        print(f"estimate n={n}: {estimates[str(n)]:.3f} s", flush=True)
    per_size = ROUTES_PER_FAMILY * len(FAMILIES) * INITIALIZATIONS
    estimates["projected_total_seconds"] = sum(
        estimates[str(n)] * per_size for n in SIZES
    )
    manifest["status"] = "estimated"
    manifest["execution"]["runtime_estimate_seconds"] = estimates
    manifest["execution"]["script_sha256"] = sha256(Path(__file__))
    manifest["execution"]["protocol_sha256"] = locked_protocol_hash(manifest)
    write_manifest(manifest)
    print(
        f"projected total: {estimates['projected_total_seconds']:.1f} s",
        flush=True,
    )


def run_full(manifest: dict[str, object]) -> None:
    started = time.perf_counter()
    manifest["status"] = "running"
    manifest["execution"]["started_at"] = utc_now()
    manifest["execution"]["script_sha256"] = sha256(Path(__file__))
    manifest["execution"]["protocol_sha256"] = locked_protocol_hash(manifest)
    write_manifest(manifest)
    rows: list[dict[str, object]] = []
    for n in SIZES:
        hamiltonian = tfim_hamiltonian(n)
        ground_energy, _ = exact_ground_state(hamiltonian)
        routes = build_routes(n)
        for route in routes:
            for initialization_index in range(INITIALIZATIONS):
                seed = BASE_SEED + 1000 * n + 37 * initialization_index
                row = analyze_instance(
                    route,
                    initialization_index,
                    seed,
                    hamiltonian,
                    ground_energy,
                    manifest,
                )
                rows.append(row)
                print(
                    f"{len(rows):03d}/180 {route.route_id} init={initialization_index}: "
                    f"grad={float(row['parameter_gradient_norm']):.2e}, "
                    f"chi={float(row['chi_ratio']):.2e}, "
                    f"dE={float(row['energy_error']):.3e}, "
                    f"event={int(row['event_primary'])}",
                    flush=True,
                )
    write_csv(rows)
    make_figure(rows)
    write_report(rows, manifest)
    manifest["status"] = "completed"
    manifest["execution"]["completed_at"] = utc_now()
    manifest["execution"]["wall_seconds"] = time.perf_counter() - started
    manifest["execution"]["software"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }
    manifest["execution"]["results_sha256"] = sha256(RESULTS_PATH)
    manifest["execution"]["figure_sha256"] = sha256(FIGURE_PDF)
    manifest["execution"]["report_sha256"] = sha256(REPORT_PATH)
    write_manifest(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered TFIM routing near-stagnation pilot."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate-only", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        help="Summary-only output directory (default: rerun/stagnation_genericity).")
    args = parser.parse_args()
    if args.output_dir is not None and not args.summarize_only:
        parser.error("--output-dir requires --summarize-only")
    return args


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    if args.estimate_only:
        run_estimate(manifest)
    elif args.summarize_only:
        summarize_saved_results(manifest, args.output_dir)
    else:
        run_full(manifest)


if __name__ == "__main__":
    main()
