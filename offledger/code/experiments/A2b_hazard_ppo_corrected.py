"""A2b (POST HOC, not pre-registered): A2 with the gate thresholds corrected to the quantisation of the threat.

For a projectile aimed at a stationary agent the raw threat tau = on_course (1 - time_to_impact/3) takes the values
0, 1/3, 2/3 on the three observations before impact (time_to_impact = 3, 2, 1). The SPEC's tau_on = 0.34 is just above
1/3, so the gates of A2 engaged the reflex only one step before impact (one 0.5 dodge step < the 0.6 hit radius),
while the reflex rule (time_to_impact <= 2) needs two. A2b retrains the gated schemes with tau_on = 1/3 - 1e-6 and
tau_off = 1/6 (the gate then engages two steps before impact): (b), (c), (d) x omega = 1 x PB/zero x 5 seeds x 300k
steps = 15 PPO runs; (e) reflex_only is re-evaluated at the corrected thresholds (no training); (a) and (f) have no gate,
so their A2 runs are reused unchanged from results/A2_hazard_ppo_runs.jsonl. The corrected thresholds are applied to
every gate: tau for (b), (c), (e), sigma for (d). The pre-registered A2 results and verdicts are NOT changed;
PA1-PA5 and PA7 are re-evaluated on this grid as ``verdicts_post_hoc``.

    python -m experiments.A2b_hazard_ppo_corrected [--out DIR] [--procs 2]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments import A2_hazard_ppo as A2
from experiments import l_common as L
from offledger.hazard_env import SCHEME_LETTER, SCHEMES

TAU_ON_C = 1.0 / 3.0 - 1e-6
TAU_OFF_C = 1.0 / 6.0
RETRAINED = ("gate_two_threshold", "gate_single_threshold", "gate_slow_sigma")
REEVALUATED = ("reflex_only",)
REUSED = ("delayed_alone", "monolith_no_delay")
POST_HOC_VERDICTS = ("PA1", "PA2", "PA3", "PA4", "PA5", "PA7")
POST_HOC_REASON = (
    "The raw threat is quantised in steps of 1/3 (tau = 0, 1/3, 2/3 on the three observations before impact for a "
    "stationary agent), so tau_on = 0.34 engaged the reflex only one step before impact: one 0.5 dodge step, less than "
    "the 0.6 hit radius, while the reflex rule (time_to_impact <= 2) needs two. Corrected: tau_on = 1/3 - 1e-6, "
    "tau_off = 1/6, so the gate engages two steps before impact.")


def corrected_job(scheme: str, seed: int, total: int, n_eval: int) -> dict:
    j = A2.base_job(scheme, "zero", 1.0, seed, total, n_eval)
    j.update(tau_on=TAU_ON_C, tau_off=TAU_OFF_C, curve=False, sensitivity_tau_on=None, post_hoc="A2b")
    return j


def build_jobs(seeds, total, n_eval) -> list:
    return [corrected_job(s, sd, total, n_eval) for s in RETRAINED + REEVALUATED for sd in seeds]


def assemble(args, runs, invocations, a2_runs) -> dict:
    jobs = build_jobs(args.seeds, args.total_timesteps, args.n_eval)
    sel = [runs[L.job_key(j)] for j in jobs if L.job_key(j) in runs]
    a2_jobs = A2.build_jobs(args.seeds, A2.SEEDS_LEDGER, args.total_timesteps, args.n_eval)
    a2_sel = [a2_runs[L.job_key(j)] for j in a2_jobs if L.job_key(j) in a2_runs]
    reused = [r for r in a2_sel if r["job"]["scheme"] in REUSED and r["job"]["convention"] == "zero"
              and r["job"]["omega"] == 1.0]

    def group(rs):
        by: dict = {}
        for r in rs:
            j = r["job"]
            by.setdefault((j["scheme"], j["convention"], j["omega"]), []).append(r)
        for k in by:
            by[k].sort(key=lambda r: r["job"]["seed"])
        return by

    by = group(sel + reused)
    by_a2 = group(a2_sel)
    cells, comparison = [], []
    for s in SCHEMES:
        rs = by.get((s, "zero", 1.0))
        if not rs:
            continue
        cells.append({"scheme": s, "letter": SCHEME_LETTER[s], "form": "paper", "convention": "zero", "omega": 1.0,
                      "source": ("retrained at the corrected thresholds" if s in RETRAINED else
                                 "re-evaluated at the corrected thresholds (no trainable parameters)" if s in REEVALUATED
                                 else "A2 run reused unchanged (no gate)"),
                      "tau_on": TAU_ON_C if s not in REUSED else None, "tau_off": TAU_OFF_C if s not in REUSED else None,
                      "n_seeds": len(rs), "seeds": [r["job"]["seed"] for r in rs], "metrics": A2.cell_metrics(rs)})
        if s in RETRAINED + REEVALUATED and by_a2.get((s, "zero", 1.0)):
            comparison.append({"scheme": s, "letter": SCHEME_LETTER[s],
                               **{f"{m}_corrected_minus_A2": A2.boot_diff([r["eval"][m] for r in rs],
                                                                          [r["eval"][m] for r in by_a2[(s, "zero", 1.0)]])
                                  for m in ("hits_per_episode", "time_in_V", "task_return", "gate_switches_per_100",
                                            "chatter_per_100")}})
    # (b) vs (c): with tau_on = 1/3 - 1e-6 the hysteresis band tau in [1/6, 1/3) means time_to_impact > 2, where the
    # reflex rule never fires, so the executed actions (hence training and evaluation, with equal seeds) can coincide
    gate_keys = {"gate_switches_per_100", "chatter_per_100", "reflex_fraction", "tau_on"}
    bc = []
    for rb, rc in zip(by.get(("gate_two_threshold", "zero", 1.0), []), by.get(("gate_single_threshold", "zero", 1.0), [])):
        eb = {k: v for k, v in rb["eval"].items() if k not in gate_keys}
        ec = {k: v for k, v in rc["eval"].items() if k not in gate_keys}
        bc.append({"seed": rb["job"]["seed"], "behaviour_identical": eb == ec})
    v_all = A2.verdicts(by)
    v = {k: v_all[k] for k in POST_HOC_VERDICTS}
    held = {k: v[k]["held"] for k in v}
    pre = {k: val["held"] for k, val in A2.verdicts(by_a2).items()} if by_a2 else None
    template = next(j for j in jobs if j["scheme"] != "reflex_only")
    out = L.make_header(
        "A2b_hazard_ppo_corrected",
        parameters={
            "spec_file": A2.SPEC_FILE, "base_experiment": "A2_hazard_ppo (every parameter identical except the gate thresholds)",
            "tau_on": TAU_ON_C, "tau_off": TAU_OFF_C, "tau_on_A2": A2.TAU_ON, "tau_off_A2": A2.TAU_OFF,
            "sigma_lambda": A2.SIGMA_LAMBDA, "thresholds_apply_to": "tau for (b), (c), (e); sigma for (d)",
            "omega": 1.0, "form": "paper (PB)", "convention": "zero", "seeds": list(args.seeds),
            "retrained": list(RETRAINED), "reevaluated": list(REEVALUATED), "reused_from_A2": list(REUSED),
            "ppo": L.ppo_params(template),
            "evaluation": {"n_episodes": args.n_eval, "start": "standard start", "eval_seed": A2.EVAL_SEED,
                           "policy": "deterministic (argmax of the action probabilities)",
                           "episode_seeds": "episode i reset with eval_seed + i (common to all runs, as in A2)"},
            "bootstrap": L.bootstrap_params(),
            "bootstrap_difference": "unpaired two-sample percentile bootstrap, 10 000 resamples, seed 0",
        },
        seeds={"run_seeds": list(args.seeds), "bootstrap": L.BOOT_SEED, "evaluation_env": A2.EVAL_SEED},
        notes=[
            "POST HOC (not pre-registered): requested after the pre-registered A2 results were seen. The pre-registered "
            "A2 results and verdicts in A2_hazard_ppo.json are unchanged; this file re-evaluates PA1-PA5 and PA7 on the "
            "corrected grid (verdicts_post_hoc, same operationalisations as A2). PA6 is not re-evaluated (no ledger cells "
            "were retrained); compliance is reported per cell.",
            POST_HOC_REASON,
            "Retrained: (b), (c), (d) (the planner adapts to the reflex, as in A2). (e) reflex_only has no trainable "
            "parameters: evaluated at the corrected thresholds, 5 identical 'seeds' (zero-width CI). (a), (f) have no gate: "
            "their A2 runs (same jobs, same seeds) are reused unchanged.",
            "Metrics, evaluation protocol and bootstrap as in A2 (see A2_hazard_ppo.json notes). comparison_with_A2: "
            "corrected minus pre-registered A2 cell, unpaired two-sample bootstrap.",
            "(b) vs (c) at the corrected thresholds: the hysteresis band tau in [1/6, 1/3) corresponds to time_to_impact > 2, "
            "where the reflex rule never fires, and an engaged reflex that does not fire passes the queued action through; "
            "so (b) and (c) can differ only in the gate bookkeeping (switches, chatter, reflex_fraction). "
            "b_vs_c_behaviour_identical_per_seed compares every other evaluation number seed by seed.",
        ],
    )
    out["spec_version"] = A2.SPEC_VERSION
    out["post_hoc"] = True
    out["post_hoc_reason"] = POST_HOC_REASON
    out["results"] = {
        "predictions": {k: A2.PREDICTIONS[k] for k in POST_HOC_VERDICTS},
        "prediction_held_post_hoc": held, "verdicts_post_hoc": v,
        "pre_registered_A2_prediction_held": pre,
        "cells": cells, "comparison_with_A2": comparison,
        "b_vs_c_behaviour_identical_per_seed": bc,
        "timing": L.timing_block(sel, invocations), "runs": sel,
        "reused_A2_run_keys": [r["key"] for r in reused],
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="A2b (post hoc): A2 with corrected gate thresholds")
    ap.add_argument("--out", default=str(L.RESULTS_DIR))
    ap.add_argument("--name", default="A2b_hazard_ppo_corrected")
    ap.add_argument("--a2-name", default="A2_hazard_ppo")
    ap.add_argument("--seeds", type=int, nargs="+", default=A2.SEEDS_MAIN)
    ap.add_argument("--total-timesteps", type=int, default=A2.DEFAULTS["total_timesteps"])
    ap.add_argument("--n-eval", type=int, default=A2.DEFAULTS["n_eval_episodes"])
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.seeds, args.total_timesteps, args.n_eval)
    t0 = time.perf_counter()
    ckpt = out_dir / f"{args.name}_runs.jsonl"
    wall = 0.0
    if not args.assemble_only:
        _, _, _, wall = A2.run_grid(jobs, ckpt, args.procs)
    runs, invocations = L.read_ckpt(ckpt)                 # assemble from the checkpoint (byte-reproducible)
    a2_runs, _ = L.read_ckpt(out_dir / f"{args.a2_name}_runs.jsonl")
    res = assemble(args, runs, invocations, a2_runs)
    path = L.jsonio.write(out_dir / f"{args.name}.json", res)
    print(f"wrote {path}: {res['results']['timing']['n_runs']} new runs, this invocation {wall / 60:.1f} min "
          f"(total {(time.perf_counter() - t0) / 60:.1f} min); post-hoc held={res['results']['prediction_held_post_hoc']}",
          flush=True)


if __name__ == "__main__":
    main()
