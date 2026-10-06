# VQC Routing Geometry

Figures and statistics for **When Expressivity Is Not Enough: Discrete
Routing Geometry in Variational Quantum Circuits**, by Yu Wang.

This package uses frozen results. No optimization runs are needed.

## Run

Verified on Linux with Python 3.12. On Windows, use WSL.

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

Outputs: `build/figures/`, `build/statistics/`, and `build/checks.json`.

## Figures

| Figure | Content | Code |
|---|---|---|
| 1 | Same-point repair | Static asset |
| 2 | Cut budget | Static asset |
| 3 | Bell capacity and access | `plots/bell.py` |
| 4 | Controlled intervention | `plots/controlled.py` |
| 5 | Heisenberg selector | `plots/heisenberg.py` |
| 6 | Routing ensemble | `plots/routing.py` |
| 7 | Candidate ranks | `plots/controlled.py` |
| 8 | TFIM selector | `plots/tfim.py` |

Six numerical figures are rebuilt; two schematics are copied.
PDF bytes may vary with fonts and platform.

## Files

- `data/`: frozen inputs and provenance.
- `plots/`: plotting and statistics.
- `reference/`: original figures and comparison tables.
- `tests/`: reproduction checks.

## Verify

```bash
python -m pytest -q
python reproduce.py check
```

Reproduction compares four tables at `atol=1e-12`, `rtol=1e-10` and checks
release hashes before and after running. All saved routing instances are
retained, including optimizer failures.

Code and data licensing is pending the author's decision.
