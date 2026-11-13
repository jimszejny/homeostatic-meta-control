"""E7b: undertrained-vs-genuine test for the cells of E7_ppo.json where PPO disagrees with the DP.

Re-trains every seed of each disagreeing (form, convention, omega) cell with a longer budget
(default 600 000 steps = 4x) and a learning-curve snapshot every 50 000 steps, with otherwise
identical hyper-parameters, seeds and evaluation. If the decision at (B,1) (and the per-battery
decisions) converge to the DP decision, the E7 disagreement was under-training, not a property of
the PPO objective.

    python -m experiments.E7b_ppo_long                  # cells with agreement < 1 in E7_ppo.json
    python -m experiments.E7b_ppo_long --cells paper max_drive 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from offledger import jsonio, ppo_utils

try:
    from experiments import E7_ppo as e7
    from experiments.common import RESULTS_DIR
except ImportError:  # pragma: no cover
    import E7_ppo as e7
    from common import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--base", default=None, help="E7_ppo.json to read (default: <out>/E7_ppo.json)")
    ap.add_argument("--cells", nargs=3, action="append", metavar=("FORM", "CONV", "OMEGA"),
                    help="explicit cell(s); default: every cell of E7_ppo.json with agreement < 1")
    ap.add_argument("--total-timesteps", type=int, default=600_000)
    ap.add_argument("--curve-every", type=int, default=50_000)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--name", default="E7b_ppo_long")
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    out_dir = Path(args.out)
    base = json.loads(Path(args.base or out_dir / "E7_ppo.json").read_text(encoding="utf-8"))
    n_eval = base["parameters"]["evaluation"]["n_episodes"]
    if args.cells:
        cells = [(f, c, float(w)) for f, c, w in args.cells]
    else:
        cells = [(a["form"], a["convention"], a["omega"]) for a in base["results"]["aggregates"]
                 if a["agreement"] < 1.0 or a["agreement_det"] < 1.0]
    seeds = base["parameters"]["seeds"]
    print(f"cells: {cells}; seeds {seeds}; {args.total_timesteps} steps", flush=True)

    jobs = [{"form": f, "convention": c, "omega": w, "seed": s, "c_r": ppo_utils.C_R,
             "total_timesteps": args.total_timesteps, "n_eval_episodes": n_eval,
             "curve_every": args.curve_every} for (f, c, w) in cells for s in seeds]
    ckpt = out_dir / f"{args.name}_runs.jsonl"
    done, _ = e7.read_ckpt(ckpt)
    t0 = time.perf_counter()
    done, n_new = e7.run_grid(jobs, done, ckpt, args.procs)
    wall = time.perf_counter() - t0

    runs = [done[e7.job_key(j)] for j in jobs]
    refs = [e7.dp_reference(f, c, w, n_eval=n_eval) for (f, c, w) in cells]
    aggs = e7.aggregate(runs, refs)
    base_aggs = {(a["form"], a["convention"], a["omega"]): a for a in base["results"]["aggregates"]}
    cmp_rows = []
    for a in aggs:
        b = base_aggs[(a["form"], a["convention"], a["omega"])]
        cmp_rows.append({
            "form": a["form"], "convention": a["convention"], "omega": a["omega"],
            "dp_comply": a["dp_comply"],
            "steps_base": base["parameters"]["ppo"]["env_steps_actually_trained"],
            "steps_long": sorted({r["timesteps_trained"] for r in runs}),
            "mean_compliance_at_full_base": b["mean_compliance_at_full"],
            "mean_compliance_at_full_long": a["mean_compliance_at_full"],
            "ci95_long": a["ci95_compliance_at_full"],
            "agreement_base": b["agreement"], "agreement_long": a["agreement"],
            "agreement_det_base": b["agreement_det"], "agreement_det_long": a["agreement_det"],
            "agreement_by_b_base": b["agreement_by_b_mean"], "agreement_by_b_long": a["agreement_by_b_mean"],
            "mean_compliance_rate_base": b["mean_compliance_rate"],
            "mean_compliance_rate_long": a["mean_compliance_rate"],
            "dp_eval_compliance_rate": a["dp_eval_compliance_rate"],
            "mean_exact_regret_B0_det_base": b["mean_exact_regret_B0_det"],
            "mean_exact_regret_B0_det_long": a["mean_exact_regret_B0_det"],
            "curve_timesteps": a["curve_timesteps"],
            "curve_mean_compliance_at_full": a["curve_mean_compliance_at_full"],
            "curve_fraction_seeds_agreeing": a["curve_fraction_seeds_agreeing"],
        })
    out = jsonio.header(
        "E7b_ppo_long",
        parameters={"cells": [{"form": f, "convention": c, "omega": w} for f, c, w in cells],
                    "seeds": seeds, "total_timesteps": args.total_timesteps,
                    "curve_every": args.curve_every, "base_file": "E7_ppo.json",
                    "same_as_E7": "all other hyper-parameters, seeds, evaluation protocol",
                    "procs": args.procs},
        notes=["Same code path as E7 (ppo_utils.train_and_evaluate) with a longer budget: tests whether "
               "the E7 disagreements disappear with more training (under-training) or persist (genuine).",
               "PPO stops after the first rollout reaching total_timesteps (4096 env steps per rollout)."],
        seeds={"run_seeds": seeds},
    )
    out["results"] = {"wall_seconds": wall, "n_runs_executed": n_new, "comparison": cmp_rows,
                      "aggregates": aggs,
                      "disagreements": [d for r in runs for d in
                                        [e7.diagnose(r, next(x for x in refs if x["form"] == r["form"]
                                                             and x["convention"] == r["convention"]
                                                             and x["omega"] == r["omega"]))] if d],
                      "runs": runs}
    path = jsonio.write(out_dir / f"{args.name}.json", out)
    print(f"wrote {path} ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
