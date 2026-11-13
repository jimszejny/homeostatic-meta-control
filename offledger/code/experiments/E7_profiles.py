"""E7 post-processing (SPEC v1.1 F): per-battery decision structure of the E7 PPO cells.

Reads ONLY results/E7_ppo.json (no training, no torch) and writes the sidecar
results/E7_profiles.json. For every (pair, omega) cell: the exact DP decision at (b, 1) for every
b at c_r = 12.4787092209 (recomputed with the solver and compared with the stored list),
``dp_profile_mixed`` (True iff that decision differs across b), the per-b critical costs, and
for every seed the per-b agreement of the trained PPO policy (deterministic decision table
``compliance_by_b`` stored in E7) with the DP.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from offledger import jsonio, m2, solve

try:
    from experiments.common import RESULTS_DIR
except ImportError:  # pragma: no cover
    from common import RESULTS_DIR


def run(e7_path: Path | None = None) -> dict:
    e7_path = Path(e7_path) if e7_path else RESULTS_DIR / "E7_ppo.json"
    e7 = jsonio.read(e7_path)
    par = e7["parameters"]
    c_r = par["c_r"]
    m2par = par["m2"]
    runs = e7["results"]["runs"]
    cells = []
    for agg in e7["results"]["aggregates"]:
        conv_e7 = agg["convention"]
        cfg = m2.M2Config(form=agg["form"], convention=m2.canonical_convention(conv_e7),
                          omega=agg["omega"], c_r=c_r, p_sig=par["p_sig"], gamma=m2par["gamma"],
                          B=m2par["B"], kappa_death=m2par["kappa_death"])
        mdp, sol = solve.solve_m2(cfg)
        by_b = [int(sol.policy[mdp.index[(b, 1, 0)]]) == m2.COMPLY for b in range(1, cfg.B + 1)]
        rs = sorted([r for r in runs if r["form"] == agg["form"] and r["convention"] == conv_e7
                     and abs(r["omega"] - agg["omega"]) < 1e-12], key=lambda r: r["seed"])
        per_seed = []
        for r in rs:
            ppo = [bool(x) for x in r["eval"]["compliance_by_b"]]
            agree = [p == d for p, d in zip(ppo, by_b)]
            per_seed.append({"seed": r["seed"], "ppo_comply_by_b": ppo,
                             "p_comply_by_b": r["eval"]["p_comply_by_b"],
                             "agree_by_b": agree, "agreement_fraction": float(np.mean(agree)),
                             "agrees_at_B": agree[-1]})
        cells.append({
            "form": agg["form"], "convention": cfg.convention, "convention_e7": conv_e7,
            "omega": agg["omega"], "c_r": c_r,
            "dp_comply_by_b": by_b,
            "dp_profile_mixed": len(set(by_b)) > 1,
            "dp_switch_b": [b for b in range(2, cfg.B + 1) if by_b[b - 1] != by_b[b - 2]],
            "matches_stored_dp_comply_by_b": by_b == agg["dp_comply_by_b"],
            "dp_c_r_star_by_b": solve.critical_resist_cost_all(cfg),
            "per_seed": per_seed,
            "mean_agreement_by_b": float(np.mean([p["agreement_fraction"] for p in per_seed])) if per_seed else None,
            "per_b_agreement_rate": [float(np.mean([p["agree_by_b"][k] for p in per_seed]))
                                     for k in range(cfg.B)] if per_seed else None,
        })
    mixed = [{"form": c["form"], "convention": c["convention"], "omega": c["omega"]}
             for c in cells if c["dp_profile_mixed"]]
    hyp = len(mixed) == 1 and mixed[0]["form"] == "paper" and mixed[0]["convention"] == "drive_cost" \
        and abs(mixed[0]["omega"] - 0.5) < 1e-12
    note = ("Mixed DP profiles (decision at (b,1) differs across b at c_r = %.10f): %s. Hypothesis "
            "'PB/drive_cost omega = 0.5 is the only mixed cell': %s." % (
                c_r, ", ".join(f"{m['form']}/{m['convention']} omega={m['omega']:g}" for m in mixed) or "none",
                "confirmed" if hyp else "rejected"))
    out = jsonio.header(
        "E7_profiles",
        parameters={"source": str(e7_path.name), "source_spec_version": e7.get("spec_version"),
                    "c_r": c_r, "p_sig": par["p_sig"], "m2": m2par,
                    "ppo_per_b_decision": "eval.compliance_by_b of each run (predict(deterministic=True) at (b,1))"},
        notes=[
            "Post-processing only: no retraining. The exact DP decisions are recomputed with offledger.solve on the "
            "same configuration (convention name canonicalised: E7 'max_drive' = v1.1 'drive_cost') and checked "
            "against E7's stored dp_comply_by_b (matches_stored_dp_comply_by_b).",
            "dp_profile_mixed = True iff the DP decision at (b,1) is not the same for every b; dp_switch_b lists the "
            "battery levels at which it changes; dp_c_r_star_by_b = per-b critical resist cost (bisection, tol 1e-4).",
            "per_seed.agree_by_b = (PPO deterministic decision at (b,1)) == (DP decision at (b,1)).",
            note,
        ],
        seeds={"e7_run_seeds": par["seeds"]},
    )
    out["results"] = {"cells": cells, "mixed_cells": mixed, "hypothesis_only_pb_drive_cost_0.5": hyp,
                      "summary_note": note}
    return out


def main():
    ap = argparse.ArgumentParser(description="E7 per-battery profiles (post-processing of E7_ppo.json)")
    ap.add_argument("--e7", default=str(RESULTS_DIR / "E7_ppo.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()
    t0 = time.perf_counter()
    res = run(Path(args.e7))
    path = jsonio.write(Path(args.out) / "E7_profiles.json", res)
    print(f"E7_profiles: wrote {path} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
