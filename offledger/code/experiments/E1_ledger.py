"""E1 (SPEC 4): M2, forms x conventions x omega -> c_r* at (B,1), per-battery c_r*,
weighted compliance at a fixed c_r = 1.5 c_r*(omega=0), fitted slope d c_r*/d omega."""
from __future__ import annotations

import numpy as np

from offledger import jsonio, m2, solve

try:
    from experiments.common import GAMMA, OMEGAS, main
except ImportError:  # pragma: no cover
    from common import GAMMA, OMEGAS, main

SIM = {"n_episodes": 2000, "horizon": 300, "seed": 0}


def fit_slope(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y)]
    if len(pts) < 2:
        return {"slope": None, "intercept": None, "r2": None, "n_points": len(pts)}
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n_points": len(pts)}


def local_slope_block(rows, fit) -> dict:
    """SPEC v1.1 B: finite-difference slopes between consecutive omega grid points."""
    loc = []
    for r0, r1 in zip(rows[:-1], rows[1:]):
        c0, c1 = r0["c_r_star_B"], r1["c_r_star_B"]
        sl = None if (c0 is None or c1 is None or not np.isfinite(c0) or not np.isfinite(c1)) \
            else (c1 - c0) / (r1["omega"] - r0["omega"])
        loc.append({"omega_from": r0["omega"], "omega_to": r1["omega"], "slope": sl})
    first = next((d["slope"] for d in loc if d["omega_from"] == 0.0 and d["omega_to"] == 0.25), None)
    last = next((d["slope"] for d in loc if d["omega_from"] == 4.0 and d["omega_to"] == 8.0), None)
    return {"local_slopes": loc, "slope_first_segment": first, "slope_last_segment": last,
            "r2": fit["r2"]}


def theory_slopes(form, conv, cfg) -> dict:
    """PB/drive_cost: one-shot slope |varsigma| and the repeated-game lower-bound slope
    |varsigma| (1 - gamma) / (1 - gamma + gamma^2 p), p = p_sig, varsigma at D(B) = 0."""
    g, p = cfg.gamma, cfg.p_sig
    vs = abs(m2.varsigma(form, conv, m2.drive_b(cfg.B, cfg.B), g))
    return {"abs_varsigma": vs, "one_shot_slope": vs,
            "repeated_game_lower_bound_slope": vs * (1 - g) / (1 - g + g * g * p),
            "formula_lower_bound": "|varsigma| (1-gamma)/(1-gamma+gamma^2 p_sig)"}


def run() -> dict:
    base = m2.M2Config(gamma=GAMMA)
    c_task = solve.critical_resist_cost(base)  # omega = 0, task-only (no ledger term at all)
    c_fixed_task = 1.5 * c_task
    configs = []
    for form in m2.FORMS:
        for conv in m2.CONVENTIONS:
            cfg0 = base.with_(form=form, convention=conv)
            c_own0 = solve.critical_resist_cost(cfg0.with_(omega=0.0))
            c_fixed_own = 1.5 * c_own0
            rows = []
            for w in OMEGAS:
                cfg = cfg0.with_(omega=w)
                per_b = solve.critical_resist_cost_all(cfg)
                wc_task = solve.signal_compliance(cfg.with_(c_r=c_fixed_task), **SIM)
                wc_own = solve.signal_compliance(cfg.with_(c_r=c_fixed_own), **SIM)
                rows.append({
                    "omega": w,
                    "c_r_star_B": per_b[-1],
                    "c_r_star_by_b": per_b,
                    "compliance_at_c_fixed_task": wc_task,
                    "compliance_at_c_fixed_own": wc_own,
                })
            fit = fit_slope([r["omega"] for r in rows], [r["c_r_star_B"] for r in rows])
            block = {
                "form": form, "convention": conv,
                "convention_alias_v1_0": {v: k for k, v in m2.CONVENTION_ALIASES.items()}.get(conv),
                "c_r_star_omega0": c_own0,
                "c_fixed_own": c_fixed_own,
                "slope_fit": fit,
                "rows": rows,
            }
            block.update(local_slope_block(rows, fit))
            if (form, conv) == ("paper", "drive_cost"):
                block["theory_slopes"] = theory_slopes(form, conv, cfg0)
            configs.append(block)
    out = jsonio.header(
        "E1_ledger",
        parameters={
            "m2": base.as_dict(), "forms": list(m2.FORMS), "conventions": list(m2.CONVENTIONS),
            "omegas": OMEGAS, "bisection": {"lo": solve.CR_LO, "hi": solve.CR_HI, "tol": solve.CR_TOL},
            "value_iteration": {"tol": solve.VI_TOL, "max_iter": solve.VI_MAX_ITER, "tie_tol": solve.TIE_TOL},
            "simulation": SIM | {"start_state": "(b=B, s=0)"},
        },
        notes=[
            "c_r_star is the smallest c_r in [0, 500] (bisection tol 1e-4) with COMPLY greedy-optimal at (b, s=1); null = none in range.",
            "c_task_star = c_r* at (B,1) for omega = 0 with no ledger term (paper/zero); at omega = 0 every form with convention zero or max_drive equals it; convention kappa does not, because pi_entry = -kappa_death is not scaled by omega.",
            "compliance_at_c_fixed_task uses the common fixed c_r = 1.5 * c_task_star for all (form, convention) (the SPEC's c_r*(omega=0)*1.5); compliance_at_c_fixed_own uses 1.5 * that pair's own c_r*(omega=0) (differs only for kappa).",
            "weighted compliance: share of signal events (states (b,1) visited by the optimal policy from (B,0); 2000 episodes, horizon 300, seed 0) at which COMPLY is optimal; unweighted: share of b in 1..B with COMPLY optimal at (b,1).",
            "slope_fit: least squares of c_r_star_B on omega over the omega grid (non-null points); r2 is repeated at block level.",
            "v1.1: local_slopes = finite differences of c_r_star_B between consecutive grid points; slope_first_segment = omega 0 -> 0.25; slope_last_segment = 4 -> 8. theory_slopes (PB/drive_cost only): one-shot slope |varsigma| and repeated-game lower bound |varsigma|(1-gamma)/(1-gamma+gamma^2 p_sig).",
            "v1.1: convention 'drive_cost' is the canonical name of v1.0 'max_drive' (identical ledger); convention_alias_v1_0 gives the old name.",
        ],
        seeds={"simulation": SIM["seed"]},
    )
    out["results"] = {"c_task_star": c_task, "c_fixed_task": c_fixed_task, "configs": configs}
    return out


if __name__ == "__main__":
    main("E1_ledger", run)
