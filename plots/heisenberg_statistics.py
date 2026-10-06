from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

BOOTSTRAP_SEED = 20260804
BOOTSTRAP_DRAWS = 200000
TIE_TOLERANCE = 1e-07


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(data_dir: Path, output_dir: Path) -> None:
    METHOD_CSV = data_dir / "methods.csv"
    CHECKPOINT_CSV = data_dir / "checkpoints.csv"
    OUTPUT_CSV = output_dir / "xxz_integration_statistics.csv"
    method_rows = read_rows(METHOD_CSV)
    checkpoint_rows = read_rows(CHECKPOINT_CSV)
    gains: dict[str, dict[str, float]] = {}
    sizes: dict[str, int] = {}
    for row in method_rows:
        instance = row["instance"]
        gains.setdefault(instance, {})[row["method"]] = float(row["gain_50"])
        sizes[instance] = int(row["n"])
    expected_random = {
        row["instance"]: float(row["random_expected_gain"]) for row in checkpoint_rows
    }
    instances = sorted(gains, key=lambda item: (sizes[item], item))
    values = {
        "no_augmentation": np.asarray(
            [gains[item]["no_augmentation"] for item in instances]
        ),
        "fixed_random": np.asarray([gains[item]["random"] for item in instances]),
        "random_pool_mean": np.asarray([expected_random[item] for item in instances]),
        "candidate_only": np.asarray([gains[item]["pauli"] for item in instances]),
        "residualized": np.asarray([gains[item]["gram"] for item in instances]),
        "oracle": np.asarray([gains[item]["oracle"] for item in instances]),
    }
    contrasts = (
        ("fixed_random", "no_augmentation"),
        ("candidate_only", "no_augmentation"),
        ("residualized", "no_augmentation"),
        ("candidate_only", "fixed_random"),
        ("residualized", "fixed_random"),
        ("candidate_only", "random_pool_mean"),
        ("residualized", "random_pool_mean"),
        ("residualized", "candidate_only"),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(instances), size=(BOOTSTRAP_DRAWS, len(instances)))
    output: list[dict[str, object]] = []
    for left, right in contrasts:
        difference = values[left] - values[right]
        bootstrap_means = np.mean(difference[indices], axis=1)
        wins = int(np.sum(difference > TIE_TOLERANCE))
        ties = int(np.sum(np.abs(difference) <= TIE_TOLERANCE))
        losses = int(np.sum(difference < -TIE_TOLERANCE))
        non_ties = wins + losses
        sign_p = (
            float(binomtest(wins, non_ties, 0.5, alternative="two-sided").pvalue)
            if non_ties
            else 1.0
        )
        output.append(
            {
                "contrast": f"{left}_minus_{right}",
                "instances": len(instances),
                "mean_difference": float(np.mean(difference)),
                "median_difference": float(np.median(difference)),
                "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "exact_sign_test_p_two_sided": sign_p,
                "tie_tolerance": TIE_TOLERANCE,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
