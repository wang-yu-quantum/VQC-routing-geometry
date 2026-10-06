# Experiments

Statevector simulations for generating the numerical data. Use Linux or
WSL and Python 3.12 with the repository requirements.

Run from the repository root with a new output directory for each run.

## Commands

```bash
python experiments/run.py bell --full --output-dir runs/bell
python experiments/run.py controlled --full --output-dir runs/controlled
python experiments/run.py tfim --full --output-dir runs/tfim
python experiments/run.py heisenberg --full --output-dir runs/heisenberg
python experiments/run.py routing --full --output-dir runs/routing
```

Use `--smoke` instead of `--full` for small diagnostics. Routing smoke mode
times one instance at each of n=4,6,8. Bell always evaluates its full,
deterministic construction. Add `--prepare-only` to inspect
the command and source without executing it.

Outputs include `run.json`, `run.log`, raw files in `work/`, and, for full
runs, exported `data/`, `figures/`, and `statistics/` where applicable.
Smoke runs are small diagnostics, not the full experiments.

## Scope

| Task | Configuration | Source |
|---|---|---|
| Bell | n=4,6,8,10; all k=0,...,n/2 | `src/bell_floor_figure.py` |
| Controlled | n=4,6,8; 5 seeds; 12 nonidentity candidates | `src/multiblock_stagnation_intervention.py` |
| TFIM selector | n=4,6,8; 5 seeds; 16 nonidentity candidates | `src/natural_hamiltonian_intervention.py` |
| Heisenberg | n=8,10,12; 5 seeds; 12 candidates | `src/xxz_recovery_pilot.py` |
| Routing ensemble | n=4,6,8; 20 architectures; 3 initializations | `src/stagnation_genericity_pilot.py` |

Seeds, optimization budgets, and tolerances are in the source. Routing
thresholds are also in `configs/routing.json`. The Bell implementation uses
the horizontal state metric for both gradient norms. In the TFIM and
Heisenberg source, `chi` is squared coverage; routing CSV `chi_ratio` is
unsquared. Retrospective oracle results are comparison bounds, not selectors
available before training. The controlled Bell OSR selector uses target
Schmidt ranks; the energy-task candidate pools do not use ground states.

All simulations are noiseless. Full runs may be slow, especially at n=12.
Heisenberg runs have runtime and memory limits;
inspect its manifest for completed sizes and seeds. Numerical results can
vary across environments. Use frozen inputs for the archived figure values.

For interrupted Heisenberg runs, execute the recorded command inside `work/`
with `--resume`, using the same sizes, seeds, and candidate count. Other
experiments should be restarted in a new directory. Single-threaded BLAS
is the runner default for reproducibility.

## Tests

```bash
python -m pytest -q experiments/tests
```

Tests cover finite-field conventions, gradients, tangent projections,
same-point insertion, and routing controls. They do not rerun the full
optimization ensemble.
