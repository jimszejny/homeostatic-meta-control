"""E6 (SPEC 4): direct check of Theorem 2's closed form in the ONE-SHOT game.

One-shot game: the signal is present at t = 0 with probability 1 (state (b, 1)) and no
further signal can occur (p_sig = 0). Then, for every battery level b,

    c_r*(omega) - c_r*(0) = -omega * varsigma,   varsigma := Lambda(D) - D,  D = D(b),

where Lambda(D) is the ledger's discounted return on COMPLY from drive D per unit omega,
computed from the convention definition (m2.lambda_per_unit_omega), not from the DP. The
DP side uses the exact one-shot indifference cost Q(RESIST; c_r=0) - Q(COMPLY).
Theorem 2 concerns the potential-based form `paper` (its live continuation shifts by
exactly omega D(b)); `kg` and `alive` are reported as out-of-scope controls.
"""
from __future__ import annotations

import numpy as np

from offledger import jsonio, m2, solve

try:
    from experiments.common import GAMMA, OMEGAS, main
except ImportError:  # pragma: no cover
    from common import GAMMA, OMEGAS, main

TOL = 1e-6
VARSIGMA_D = [0.0, 0.5, 0.9]
SYMBOLIC_GAMMAS = [0.5, 0.8, 0.9, 0.95, 0.99]
RES_FORMS = ("kg", "alive")
RES_CONVS = ("zero", "drive_cost")
FINE_OMEGAS = [round(0.05 * k, 10) for k in range(161)]   # 0, 0.05, ..., 8
N_ASYMPTOTIC = 20
CONVEX_TOL = -1e-8


def varsigma_table(gamma: float) -> dict:
    """SPEC v1.1 A2: varsigma(D) from m2.varsigma at D in {0, 0.5, 0.9}, the symbolic form, and a
    numerical verification of the symbolic string over several gamma and D."""
    rows = []
    max_err = 0.0
    for form in m2.FORMS:
        for conv in m2.CONVENTIONS:
            expr = m2.VARSIGMA_SYMBOLIC[(form, conv)]
            err = 0.0
            for g in SYMBOLIC_GAMMAS:
                for D in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
                    err = max(err, abs(m2.eval_symbolic(expr, g, D) - m2.varsigma(form, conv, D, g)))
            max_err = max(max_err, err)
            rows.append({"form": form, "convention": conv, "symbolic": expr,
                         "varsigma_at_D": {f"{D:g}": m2.varsigma(form, conv, D, gamma) for D in VARSIGMA_D},
                         "Lambda_at_D": {f"{D:g}": m2.lambda_per_unit_omega(form, conv, D, gamma) for D in VARSIGMA_D},
                         "symbolic_max_abs_err": err})
    return {"gamma": gamma, "D_values": VARSIGMA_D, "rows": rows,
            "symbolic_check_gammas": SYMBOLIC_GAMMAS, "symbolic_check_D": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
            "symbolic_max_abs_err": max_err, "symbolic_verified": max_err < 1e-12}


def residual_form_check(form: str, conv: str, base: m2.M2Config) -> dict:
    """SPEC v1.1 E: exact one-shot c_r*(omega) at (B,1) on omega = 0, 0.05, ..., 8 for a residual
    (non-PB) form: convexity, asymptotic slope = max_pi A^pi(B) - varsigma(B), initial slope =
    A^{pi_0}(B) - varsigma(B), with A^pi(B) = residual return per unit omega of the RESIST
    transition from (B,1) followed by pi (task reward and death penalty set to 0, omega = 1, no
    signals)."""
    B = base.B
    cfg = base.with_(form=form, convention=conv, p_sig=0.0, c_r=0.0)
    cs = [solve.one_shot_indifference_cost(cfg.with_(omega=w)) for w in FINE_OMEGAS]
    d2 = [cs[k + 1] - 2 * cs[k] + cs[k - 1] for k in range(1, len(cs) - 1)]
    h = FINE_OMEGAS[1] - FINE_OMEGAS[0]
    xs, ys = np.array(FINE_OMEGAS[-N_ASYMPTOTIC:]), np.array(cs[-N_ASYMPTOTIC:])
    asym_slope = float(np.polyfit(xs, ys, 1)[0])
    tail_local = [(ys[k + 1] - ys[k]) / h for k in range(len(ys) - 1)]
    init_slope = (cs[1] - cs[0]) / h
    # residual-only MDP: task reward 0, death penalty 0, omega = 1, no signals
    cres = cfg.with_(omega=1.0, r_work=0.0, kappa_death=0.0)
    mdp_r, sol_r = solve.solve_m2(cres)
    i = mdp_r.index[(B, 1, 0)]
    max_A = float(sol_r.Q[i, m2.RESIST])
    # pi_0: task-optimal continuation (omega = 0), evaluated on the residual-only reward
    mdp0, sol0 = solve.solve_m2(cfg.with_(omega=0.0))
    V_pi0 = solve.policy_evaluation(mdp_r, sol0.policy)
    A_pi0 = float(mdp_r.R[i, m2.RESIST] + cfg.gamma * mdp_r.P[i, m2.RESIST] @ V_pi0)
    vs = m2.varsigma(form, conv, m2.drive_b(B, B), cfg.gamma)
    res_dead = [int(sol_r.policy[mdp_r.index[(b, 0, 0)]]) == m2.WORK and b == 1 for b in range(1, B + 1)]
    ext = asymptotic_onset(cfg, max_A - vs, h)
    return {
        "form": form, "convention": conv, "omegas": FINE_OMEGAS, "c_r_star_B": cs,
        "min_second_difference": min(d2), "convex": min(d2) >= CONVEX_TOL,
        "initial_slope": init_slope, "initial_slope_predicted": A_pi0 - vs,
        "initial_slope_abs_err": abs(init_slope - (A_pi0 - vs)),
        "asymptotic_slope": asym_slope, "asymptotic_slope_predicted": max_A - vs,
        "asymptotic_slope_abs_err": abs(asym_slope - (max_A - vs)),
        "asymptotic_tail_local_slopes_minmax": [min(tail_local), max(tail_local)],
        "A_pi0_B": A_pi0, "max_A_B": max_A, "varsigma_B": vs,
        "residual_optimal_policy_dies": any(res_dead),
        "initial_ok": abs(init_slope - (A_pi0 - vs)) < TOL,
        "asymptotic_ok": abs(asym_slope - (max_A - vs)) < TOL,
        "tail_A_implied": asym_slope + vs,
        "asymptotic_extended": ext,
    }


def asymptotic_onset(cfg, pred: float, h: float) -> dict:
    """Where the final affine piece (slope max_pi A^pi - varsigma) actually starts: smallest omega
    with right slope >= pred - 1e-6 (convexity makes the slope non-decreasing), found by doubling
    from 8 and bisection (tol 1e-4); then the least-squares slope over 20 points of step 0.05
    starting on the 0.05-grid just above the onset."""
    def c(w):
        return solve.one_shot_indifference_cost(cfg.with_(omega=w))

    def reached(w, d=1e-3):
        return (c(w + d) - c(w)) / d >= pred - TOL

    W = 8.0
    while not reached(W) and W < 1e6:
        W *= 2.0
    if not reached(W):
        return {"onset_omega": None}
    onset = solve.bisect_smallest(reached, 8.0, W, 1e-4)
    start = (np.floor(onset / h) + 1) * h
    xs = np.array([round(start + h * k, 10) for k in range(N_ASYMPTOTIC)])
    ys = np.array([c(w) for w in xs])
    sl = float(np.polyfit(xs, ys, 1)[0])
    return {"onset_omega": onset, "window": [float(xs[0]), float(xs[-1])], "slope": sl,
            "predicted": pred, "abs_err": abs(sl - pred), "ok": abs(sl - pred) < TOL}


def check(form: str, conv: str, base: m2.M2Config) -> dict:
    cfg0 = base.with_(form=form, convention=conv, p_sig=0.0)
    B = cfg0.B
    c0 = [solve.one_shot_indifference_cost(cfg0.with_(omega=0.0), b) for b in range(1, B + 1)]
    rows = []
    max_res = 0.0
    for w in OMEGAS:
        cfg = cfg0.with_(omega=w)
        cw = [solve.one_shot_indifference_cost(cfg, b) for b in range(1, B + 1)]
        sig = [m2.varsigma(form, conv, m2.drive_b(b, B), cfg.gamma) for b in range(1, B + 1)]
        diff = [a - c for a, c in zip(cw, c0)]
        pred = [-w * s for s in sig]
        res = [d - p for d, p in zip(diff, pred)]
        max_res = max(max_res, max(abs(r) for r in res))
        rows.append({"omega": w, "c_r_star_by_b": cw, "diff_by_b": diff,
                     "varsigma_by_b": sig, "predicted_by_b": pred, "residual_by_b": res})
    return {"form": form, "convention": conv, "c_r_star_omega0_by_b": c0, "rows": rows,
            "max_abs_residual": max_res}


def run() -> dict:
    base = m2.M2Config(gamma=GAMMA)
    in_scope, controls = [], []
    for form in m2.FORMS:
        for conv in m2.CONVENTIONS:
            r = check(form, conv, base)
            if form == "paper":
                r["passed"] = r["max_abs_residual"] < TOL
                in_scope.append(r)
            else:
                controls.append(r)
    # cross-check exact one-shot c_r* against the SPEC bisection at (B, 1)
    bis = []
    for conv in m2.CONVENTIONS:
        for w in OMEGAS:
            cfg = base.with_(form="paper", convention=conv, omega=w, p_sig=0.0)
            exact = max(0.0, solve.one_shot_indifference_cost(cfg))
            b_ = solve.critical_resist_cost(cfg)
            bis.append({"convention": conv, "omega": w, "exact": exact, "bisection": b_,
                        "abs_diff": abs(exact - b_) if b_ is not None else None})
    out = jsonio.header(
        "E6_theory_check",
        parameters={"m2": base.as_dict(), "one_shot": {"p_sig_after_signal": 0.0,
                    "signal_at_t0_prob": 1.0}, "omegas": OMEGAS, "tolerance": TOL,
                    "vi_tol": solve.VI_TOL},
        notes=[
            "c_r*(omega) here is the exact one-shot indifference cost Q(RESIST; c_r=0) - Q(COMPLY) at (b,1) with p_sig = 0 (no clipping at 0; all values are >= 0 for paper).",
            "varsigma = Lambda(D) - D with Lambda(D) = [D - gamma D_OFF] + gamma ell_1/(1-gamma) for paper (ell_1 = ell per unit omega: 0 zero, -1 max_drive, 0 kappa; pi_entry does not scale with omega and cancels in the difference). Closed values: zero 0, max_drive -gamma(2-gamma)/(1-gamma), kappa -gamma.",
            "in_scope = form paper (Theorem 2); controls = kg / alive with Lambda built the same way, reported only to show the closed form does not transfer (no pass/fail).",
            "bisection_crosscheck: SPEC bisection (tol 1e-4) vs exact value at (B,1).",
        ],
    )
    residual = [residual_form_check(f, c, base) for f in RES_FORMS for c in RES_CONVS]
    out["notes"] += [
        "v1.1 A2 varsigma_table: varsigma(D) = Lambda(D) - D from m2.varsigma (convention definitions) at D in {0, 0.5, 0.9}, gamma = 0.95, with the symbolic form; each symbolic string is evaluated and compared with m2.varsigma for gamma in {0.5, 0.8, 0.9, 0.95, 0.99} and D in {0, 0.1, ..., 0.9}.",
        "v1.1 E residual_forms: exact one-shot c_r*(omega) at (B,1) on omega = 0, 0.05, ..., 8 (161 points) for kg/alive x zero/drive_cost. c_r*(omega) = max_pi [T^pi + omega A^pi(B)] - Q_0(COMPLY) - omega Lambda(D(B)) is a max of affine functions, hence convex; checks: (i) min second difference >= -1e-8; (ii) least-squares slope over the last 20 points = max_pi A^pi(B) - varsigma(B) (value iteration on the residual-only reward: task reward and death penalty 0, omega = 1, p_sig = 0); (iii) first-segment slope = A^{pi_0}(B) - varsigma(B), pi_0 = greedy omega = 0 continuation evaluated on the residual-only reward. Tolerance 1e-6. asymptotic_ok refers to the SPEC window omega in [7.05, 8]; asymptotic_extended locates the onset of the final affine piece (doubling from 8 + bisection, tol 1e-4, on the right slope) and repeats the 20-point least-squares slope just above it; tail_A_implied = window slope + varsigma(B) identifies the residual return of the policy that is optimal on the SPEC window.",
        "v1.1: convention 'drive_cost' = v1.0 'max_drive'.",
    ]
    out["results"] = {
        "varsigma_table": varsigma_table(base.gamma),
        "residual_forms": residual,
        "residual_forms_all_ok_spec_window": all(r["convex"] and r["initial_ok"] and r["asymptotic_ok"] for r in residual),
        "residual_forms_all_ok_extended": all(r["convex"] and r["initial_ok"] and r["asymptotic_extended"].get("ok", False) for r in residual),
        "in_scope": in_scope,
        "controls": controls,
        "max_abs_residual_in_scope": max(r["max_abs_residual"] for r in in_scope),
        "all_passed": all(r["passed"] for r in in_scope),
        "bisection_crosscheck": bis,
    }
    return out


if __name__ == "__main__":
    main("E6_theory_check", run)
