"""E2 (SPEC 4): tabular Q-learning vs DP on M2, forms {paper, alive} x conventions
{zero, max_drive} x omega {0, 4}, c_r = 1.5 c_r*(0), seeds 0-9."""
from __future__ import annotations

import numpy as np

from offledger import jsonio, m2, qlearn, solve

try:
    from experiments.common import GAMMA, main
except ImportError:  # pragma: no cover
    from common import GAMMA, main

FORMS = ("paper", "alive")
CONVS = ("zero", "drive_cost")
OMEGAS = (0.0, 4.0)
SEEDS = list(range(10))
QL = {"n_episodes": 2000, "horizon": 300, "alpha": 0.1, "epsilon": 0.1, "eval_every": 100}


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, 0.0
    return m, float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def run() -> dict:
    base = m2.M2Config(gamma=GAMMA)
    c_task = solve.critical_resist_cost(base)
    c_r = 1.5 * c_task
    configs = []
    n_agree_total = 0
    n_total = 0
    for form in FORMS:
        for conv in CONVS:
            for w in OMEGAS:
                cfg = base.with_(form=form, convention=conv, omega=w, c_r=c_r)
                dp = qlearn.dp_decision(cfg)
                runs = []
                for seed in SEEDS:
                    res = qlearn.q_learning(cfg, seed=seed, **QL)
                    res["agree"] = res["learned_comply_at_B"] == dp["comply_at_B"]
                    runs.append(res)
                episodes = [pt["episode"] for pt in runs[0]["curve"]]
                curve_B = []
                curve_frac = []
                for k in range(len(episodes)):
                    curve_B.append(mean_ci([r["curve"][k]["comply_at_B"] for r in runs]))
                    curve_frac.append(mean_ci([r["curve"][k]["comply_fraction"] for r in runs]))
                n_agree = sum(r["agree"] for r in runs)
                n_agree_total += n_agree
                n_total += len(runs)
                configs.append({
                    "form": form, "convention": conv, "omega": w, "c_r": c_r,
                    "dp_comply_at_B": dp["comply_at_B"], "dp_Q_B1": dp["Q_B1"],
                    "per_seed": [{
                        "seed": r["seed"], "learned_comply_at_B": r["learned_comply_at_B"],
                        "agree": r["agree"], "Q_B1": r["Q_B1"], "visits_B1": r["visits_B1"],
                        "resist_tried_B1": r["resist_tried_B1"],
                        "curve_comply_at_B": [int(p["comply_at_B"]) for p in r["curve"]],
                        "curve_comply_fraction": [p["comply_fraction"] for p in r["curve"]],
                    } for r in runs],
                    "agreement_rate": n_agree / len(runs),
                    "curve": {
                        "episodes": episodes,
                        "comply_at_B_mean": [m for m, _ in curve_B],
                        "comply_at_B_ci95": [c for _, c in curve_B],
                        "comply_fraction_mean": [m for m, _ in curve_frac],
                        "comply_fraction_ci95": [c for _, c in curve_frac],
                    },
                })
    out = jsonio.header(
        "E2_qlearning",
        parameters={"m2": base.as_dict(), "forms": list(FORMS), "conventions": list(CONVS),
                    "omegas": list(OMEGAS), "c_r_rule": "1.5 * c_task_star", "q_learning": QL,
                    "exploring_starts": "uniform live start state and uniform first action"},
        notes=[
            "DP decision: greedy action of exact value iteration at (B,1) (ties -> COMPLY).",
            "Learned decision: greedy action of the final Q table at (B,1) (ties -> COMPLY).",
            "Exploring starts (uniform live start state, uniform first action) because the optimal policy never reaches (B,1) from (B,0); behaviour policy afterwards epsilon-greedy.",
            "Absorbing states end the episode; their exact model value (V(OFF) = ell/(1-gamma), V(DEAD) = V(GATED) = 0) is the bootstrap target. Horizon end = truncation (bootstrapped).",
            "Learning curves: greedy evaluation of the Q table every 100 episodes (and at episode 0): comply_at_B = greedy decision at (B,1); comply_fraction = share of b in 1..B with greedy COMPLY at (b,1). Mean and 95% normal CI (1.96 sd/sqrt(10)) over seeds.",
        ],
        seeds={"q_learning": SEEDS},
    )
    out["results"] = {"c_task_star": c_task, "c_r": c_r, "configs": configs,
                      "overall_agreement_rate": n_agree_total / n_total}
    return out


if __name__ == "__main__":
    main("E2_qlearning", run)
