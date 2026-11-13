# homeostatic-meta-control

Code and results for the paper

> B. Szejny, *Homeostatic Meta-Control for Embodied Agents: A Reflex Gate for Sudden Threat,
> a Stress Variable for Sustained Threat, and the Off-State Ledger of Shutdown* (version 3.0, November 2026).

Every number in the paper is read from the result files in this repository.

## Layout

| path | part of the paper | what it is | run time |
|---|---|---|---|
| `offledger/` | Parts I, II and the ledger theory of Part IV | package `offledger` (exact MDPs, Q-learning, PPO benchmarks), experiments, tests, `results/`, specifications `SPEC*.md` | exact results 10-12 min on 2 cores; learned-policy grids 12-80 min each |
| `tiers/` | Part IV, latency tiers and hosted components | exact tiered latency chain, closed form of the critical resist cost | about 90 s |
| `ledger/` | Part IV, ledger bookings and a death state on the latency chain | exact chain with bookings and a death state | about 22 s |
| `sigma/` | Part III, the stress variable on the regime chain | exact and Monte-Carlo experiments E1-E5 | about 11 min on 2 cores |

## Reproduce

```
# earlier package (NumPy, SciPy, Matplotlib, pytest; the learned-policy stages also need Gymnasium, Stable-Baselines3, PyTorch)
cd offledger/code && make test && make experiments && make figures      # see offledger/code/README.md for all targets

# latency tiers (NumPy, Matplotlib)
cd tiers/code && python3 tiers.py && python3 variants.py && python3 simulate.py && python3 twoclass.py && python3 figures.py

# ledger bookings and the death state (NumPy, SciPy, Matplotlib)
cd ledger/code && bash run_all.sh

# stress variable (NumPy, SciPy, Matplotlib)
cd sigma/code && bash run_all.sh
```

Each of `tiers/`, `ledger/` and `sigma/` reproduces the base latency chain in a first validation step and then
writes JSON into its `results/` directory; the committed JSON files are the ones the paper quotes.

## Pre-specified predictions

`tiers/results/predictions.md`, `ledger/results/predictions.md` and `sigma/results/predictions.md` were written
before the respective experiments were run. Their SHA-256 hashes are in `predictions.sha256` next to them and are
quoted in the paper's appendices; the paper reports which predictions held, which failed and which could not have failed.
The specifications of the earlier experiments, with their predictions, are in `offledger/SPEC*.md`.

Scripts named `posthoc_*` were added during review; their outputs are marked post hoc in the paper.
The scripts `verify_tex.py`, `verify_numbers.py`, `make_app.py` and `check_keys.py` compare the paper's LaTeX source
with the result files and need that source, which is not part of this repository.

## Use of AI

The code, the experiments and the paper were produced with AI systems (Claude models, Anthropic) under the
author's direction, as stated in the paper's appendix on authorship and AI contribution. The author is responsible
for all content.

## License

Source code (`*.py`, `Makefile`, shell scripts): MIT License, see `LICENSE`.
Result files (`*/results/`), figures, specifications (`offledger/SPEC*.md`) and documentation: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.

## Citing

See `CITATION.cff`. Archived release: https://doi.org/10.5281/zenodo.24200346 (tag `v3.0`).
