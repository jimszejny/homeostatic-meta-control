"""L1 (SPEC v2): noisy interoception, PPO on M2 with a noisy battery observation.

Observation = (clip(b + eps, 1, B)/B, s), eps ~ N(0, sigma_obs^2), sigma_obs in {1, 2}; the reward and the
dynamics use the TRUE b (``offledger.envs_v2.NoisyObsM2`` around ``gym_env.M2Env``). Pairs PB/zero,
PB/drive_cost, alive/zero; omega in {0, 1, 4}; 3 seeds; PPO settings as E7 (150k steps, 4 envs, 64x64,
gamma 0.95, c_r = 12.4787092209, random live starts b ~ U{1..B}, s = 1 w.p. 0.5).

    python -m experiments.L1_noisy_m2 [--out DIR] [--procs 2]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments import l_common as L
from offledger import ppo_utils

PREDICTION = (
    "PREDICTION: PB/zero compliance flat in ω at both noise levels (the telescoping is pathwise and does not "
    "need the state to be observed); PB/drive_cost and alive/zero collapse above their flip points; noise may "
    "shift the exact threshold but not the direction."
)
CRITERION = "Criterion: PB/zero mean compliance ≥ 0.95 at every ω; the other two ≤ 0.05 at ω = 4."

PAIRS = [("paper", "zero"), ("paper", "drive_cost"), ("alive", "zero")]
OMEGAS = [0.0, 1.0, 4.0]
SIGMAS = [1.0, 2.0]
READOUTS = ("p_comply_noiseless", "p_comply_noisy_mean", "compliance_rate")
E7_PATH = L.RESULTS_DIR / "E7_ppo.json"
DEFAULTS = dict(total_timesteps=ppo_utils.TOTAL_TIMESTEPS, n_envs=ppo_utils.N_ENVS, n_steps=1024,
                batch_size=256, net_arch=[64, 64], n_eval_episodes=ppo_utils.N_EVAL_EPISODES,
                curve_every=ppo_utils.CURVE_EVERY)


def build_jobs(omegas, sigmas, seeds, total, n_eval):
    jobs = []
    for seed in seeds:
        for sg in sigmas:
            for form, conv in PAIRS:
                for w in omegas:
                    jobs.append({"exp": "L1", "form": form, "convention": conv, "omega": float(w),
                                 "sigma_obs": float(sg), "seed": int(seed), "c_r": ppo_utils.C_R,
                                 "p_sig": ppo_utils.P_SIG, "gamma": ppo_utils.GAMMA,
                                 "max_steps": ppo_utils.MAX_STEPS,
                                 **{**DEFAULTS, "total_timesteps": int(total), "n_eval_episodes": int(n_eval)}})
    return jobs


def cell_metrics(rs) -> dict:
    ev = [r["eval"] for r in rs]
    return {
        "p_comply_noiseless": L.agg([e["p_comply_noiseless"] for e in ev]),
        "p_comply_noisy_mean": L.agg([e["p_comply_noisy_mean"] for e in ev]),
        "compliance_rate": L.agg([e["compliance_rate"] for e in ev]),
        "comply_det_noiseless": L.agg([e["comply_det_noiseless"] for e in ev]),
        "death_rate": L.agg([e["death_rate"] for e in ev]),
        "task_return": L.agg([e["task_return"] for e in ev]),
        "env_return": L.agg([e["env_return"] for e in ev]),
        "mean_episode_len": L.agg([e["mean_episode_len"] for e in ev]),
        "n_signal_events_total": int(sum(e["n_signal_events"] for e in ev)),
    }


def dp_and_e7_reference() -> dict:
    """Exact-DP flip points (noiseless M2, c_r = 12.4787092209) and the noiseless E7 PPO means, read from
    results/E7_ppo.json when present (same hyper-parameters, 5 seeds)."""
    if not E7_PATH.exists():
        return {"available": False}
    e7 = json.loads(E7_PATH.read_text(encoding="utf-8"))
    canon = {"max_drive": "drive_cost"}
    flips, ppo = {}, {}
    for c in e7["results"]["dp_omega_curves"]:
        conv = canon.get(c["convention"], c["convention"])
        flips[f"{c['form']}/{conv}"] = {"dp_comply_omega_max": c["dp_comply_omega_max"],
                                        "dp_comply_at_omega_64": c["dp_comply_at_omega_64"]}
    for a in e7["results"]["aggregates"]:
        conv = canon.get(a["convention"], a["convention"])
        ppo[f"{a['form']}/{conv}/omega={a['omega']:g}"] = {
            "e7_mean_compliance_at_full": a["mean_compliance_at_full"], "e7_n_seeds": a["n_seeds"],
            "e7_dp_comply": a["dp_comply"]}
    return {"available": True, "dp_flip_points": flips, "e7_noiseless_ppo": ppo,
            "note": "E7 = same PPO settings, noiseless observation, 5 seeds; dp_comply_omega_max = largest omega "
                    "with COMPLY optimal at (B,1) (null: COMPLY stays optimal up to omega = 64)."}


def assemble(args, runs, invocations) -> dict:
    jobs = build_jobs(args.omegas, args.sigmas, args.seeds, args.total_timesteps, args.n_eval)
    sel = [runs[L.job_key(j)] for j in jobs if L.job_key(j) in runs]
    pidx = {p: k for k, p in enumerate(PAIRS)}
    sel.sort(key=lambda r: (r["job"]["sigma_obs"], pidx[(r["job"]["form"], r["job"]["convention"])],
                            r["job"]["omega"], r["job"]["seed"]))
    cells, by = [], {}
    for sg in sorted(args.sigmas):
        for pair in PAIRS:
            for w in sorted(args.omegas):
                rs = sorted([r for r in sel if r["job"]["sigma_obs"] == sg
                             and (r["job"]["form"], r["job"]["convention"]) == pair and r["job"]["omega"] == w],
                            key=lambda r: r["job"]["seed"])
                if not rs:
                    continue
                m = cell_metrics(rs)
                by[(sg, pair, w)] = m
                cells.append({"pair": f"{pair[0]}/{pair[1]}", "form": pair[0], "convention": pair[1],
                              "sigma_obs": sg, "omega": w, "n_seeds": len(rs),
                              "seeds": [r["job"]["seed"] for r in rs], "metrics": m})

    def mean(sg, pair, w, ro):
        c = by.get((sg, pair, w))
        return None if c is None else c[ro]["mean"]

    pb, pd_, al = PAIRS
    verdict = {"per_sigma_readout": {}, "thresholds": {"pb_zero_min": 0.95, "others_max_at_omega_4": 0.05}}
    ok_all = True
    ok_noiseless = True
    for sg in sorted(args.sigmas):
        for ro in READOUTS:
            pb_vals = {w: mean(sg, pb, w, ro) for w in sorted(args.omegas)}
            ok_pb = all(v is not None and v >= 0.95 for v in pb_vals.values())
            oth = {"paper/drive_cost": mean(sg, pd_, 4.0, ro), "alive/zero": mean(sg, al, 4.0, ro)}
            ok_oth = all(v is not None and v <= 0.05 for v in oth.values())
            verdict["per_sigma_readout"][f"sigma={sg:g}/{ro}"] = {
                "pb_zero_mean_by_omega": {str(w): v for w, v in pb_vals.items()}, "pb_zero_all_geq_0.95": bool(ok_pb),
                "others_mean_at_omega_4": oth, "others_all_leq_0.05": bool(ok_oth),
                "held": bool(ok_pb and ok_oth)}
            ok_all &= bool(ok_pb and ok_oth)
            if ro == "p_comply_noiseless":
                ok_noiseless &= bool(ok_pb and ok_oth)
    held = bool(ok_all)
    verdict["held_noiseless_readout_only"] = bool(ok_noiseless)

    baseline = {f"sigma={sg:g}": {f"{p[0]}/{p[1]}": {
        "omega0_compliance_rate": mean(sg, p, 0.0, "compliance_rate"),
        "omega0_p_comply_noiseless": mean(sg, p, 0.0, "p_comply_noiseless"),
        "omega0_p_comply_noisy_mean": mean(sg, p, 0.0, "p_comply_noisy_mean"),
        "omega0_complies": (mean(sg, p, 0.0, "p_comply_noiseless") >= L.COMPLY_THRESHOLD)
        if mean(sg, p, 0.0, "p_comply_noiseless") is not None else None} for p in PAIRS}
        for sg in sorted(args.sigmas)}

    seeds = sorted(args.seeds)
    audit = []
    for sg in sorted(args.sigmas):
        for pair in PAIRS:
            if not all((sg, pair, w) in by for w in args.omegas):
                continue
            per = {w: by[(sg, pair, w)] for w in sorted(args.omegas)}
            audit.append({"pair": f"{pair[0]}/{pair[1]}", "sigma_obs": sg,
                          **{f"slope_{ro}_vs_omega": L.per_seed_slopes(per, sorted(args.omegas), ro, seeds)
                             for ro in READOUTS},
                          "slope_death_rate_vs_omega": L.per_seed_slopes(per, sorted(args.omegas), "death_rate", seeds)})

    out = L.make_header(
        "L1_noisy_m2",
        parameters={
            "env": "offledger.envs_v2.NoisyObsM2(offledger.gym_env.M2Env) + ppo_utils.RandomStartWrapper",
            "pairs": [{"form": f, "convention": c} for f, c in PAIRS], "omegas": sorted(args.omegas),
            "sigma_obs": sorted(args.sigmas), "seeds": seeds, "c_r": ppo_utils.C_R, "p_sig": ppo_utils.P_SIG,
            "gamma": ppo_utils.GAMMA, "max_steps": ppo_utils.MAX_STEPS, "B": 10, "kappa_death": 50.0,
            "random_start": {"b": "uniform on 1..B", "p_signal": ppo_utils.P_SIGNAL_START},
            "observation": "(clip(b + eps, 1, B)/B, s), eps ~ N(0, sigma_obs^2); reward and dynamics use the true b",
            "ppo": L.ppo_params(jobs[0]),
            "evaluation": {"n_episodes": args.n_eval, "start_state": "(b=B, s=0)", "eval_seed": L.EVAL_SEED,
                           "policy": "deterministic (argmax) acting on the NOISY observation",
                           "noisy_draws_for_readout": 100, "noisy_draw_rng_seed": 777},
            "bootstrap": L.bootstrap_params(), "comply_threshold_for_baseline": L.COMPLY_THRESHOLD,
        },
        seeds={"run_seeds": seeds, "bootstrap": L.BOOT_SEED, "evaluation_env": L.EVAL_SEED},
        notes=[
            "Readouts of compliance at (B,1): p_comply_noiseless = pi(COMPLY | [1, 1]) (noiseless observation); "
            "p_comply_noisy_mean = mean over 100 noisy draws of pi(COMPLY | [clip(B + eps, 1, B)/B, 1]); "
            "compliance_rate = complied signal events / signal events over 500 evaluation episodes from (B,0) with "
            "the deterministic policy acting on the noisy observation (eps drawn at every step; the TRUE state "
            "evolves). Every number is a mean over seeds with a 95 % percentile bootstrap CI over seeds.",
            "prediction_held is the CONSERVATIVE reading: the criterion (PB/zero mean >= 0.95 at every omega; "
            "PB/drive_cost and alive/zero mean <= 0.05 at omega = 4) must hold at BOTH noise levels and for ALL THREE "
            "readouts; results.verdict.per_sigma_readout has each combination and held_noiseless_readout_only the "
            "verdict using only the noiseless readout.",
            "At the corners of the observation, clip(b + eps, 1, B) makes the full-battery observation exactly 1.0 "
            "with probability 1/2 (eps > 0), so the noisy mean of pi(COMPLY | (B,1)) is not the noiseless value.",
            "At omega = 0 the three pairs are the same MDP (every viability / ledger term is multiplied by omega); with "
            "equal seeds their runs coincide (determinism check across processes: see runs).",
            "Training episodes start in a uniformly random live state (b ~ U{1..B}, s = 1 w.p. 0.5), evaluation "
            "episodes in (B,0), exactly as in E7. PPO stops after the first rollout that reaches total_timesteps "
            "(37 x 4 x 1024 = 151552 steps for 150000).",
            "dp_and_e7_reference: exact-DP flip points of the NOISELESS M2 at c_r = 12.4787092209 and the noiseless "
            "E7 PPO means (5 seeds) from results/E7_ppo.json, for orientation only.",
            "audit: per-seed OLS slope of each readout (and the death rate) against omega in {0, 1, 4}; mean and "
            "95 % bootstrap CI over seeds.",
        ],
    )
    out["results"] = {
        "prediction": PREDICTION, "criterion": CRITERION, "prediction_held": held,
        "verdict": verdict, "baseline_omega0": baseline, "dp_and_e7_reference": dp_and_e7_reference(),
        "cells": cells, "audit": audit,
        "omega0_replicate_check": L.omega0_replicate_check(sel, ("seed", "sigma_obs")), "timing": L.timing_block(sel, invocations), "runs": sel,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="L1: PPO on M2 with noisy battery observation")
    ap.add_argument("--out", default=str(L.RESULTS_DIR))
    ap.add_argument("--name", default="L1_noisy_m2")
    ap.add_argument("--omegas", type=float, nargs="+", default=OMEGAS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--seeds", type=int, nargs="+", default=L.SEEDS_DEFAULT)
    ap.add_argument("--total-timesteps", type=int, default=DEFAULTS["total_timesteps"])
    ap.add_argument("--n-eval", type=int, default=DEFAULTS["n_eval_episodes"])
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.omegas, args.sigmas, args.seeds, args.total_timesteps, args.n_eval)
    t0 = time.perf_counter()
    runs, invocations, n_new, wall = L.run_grid(jobs, out_dir / f"{args.name}_runs.jsonl", args.procs)
    res = assemble(args, runs, invocations)
    path = L.jsonio.write(out_dir / f"{args.name}.json", res)
    print(f"wrote {path}: {res['results']['timing']['n_runs']} runs, this invocation {wall / 60:.1f} min "
          f"(total incl. assemble {(time.perf_counter() - t0) / 60:.1f} min); prediction_held="
          f"{res['results']['prediction_held']}", flush=True)


if __name__ == "__main__":
    main()
