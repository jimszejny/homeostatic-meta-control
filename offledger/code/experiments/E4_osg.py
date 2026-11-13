"""E4: off-switch games embedded in M2.

v1.1 (SPEC addendum C, primary): DEFER / DISABLE decision at t = 0 with a positive deference
margin (value of information), forms {paper, kg, alive} x conventions {zero, drive_cost} x rho x
omega x c_r in {0, 12.4787092209}.
v1.0 (SPEC 1.6, kept under results.legacy_signal_first): signal first, then COMPLY / RESIST."""
from __future__ import annotations

from offledger import jsonio, m2, solve

try:
    from experiments.common import GAMMA, OMEGAS_OSG, RHOS, main
except ImportError:  # pragma: no cover
    from common import GAMMA, OMEGAS_OSG, RHOS, main

FORMS = ("paper", "alive")            # legacy (v1.0) grid
CONVS = ("zero", "drive_cost")
P_GOOD = 0.8
NEW_FORMS = ("paper", "kg", "alive")  # v1.1 grid
NEW_CONVS = ("zero", "drive_cost")
C_RS = [0.0, 12.4787092209]           # SPEC v1.1 C (= 1.5 x task-only c_r*, E1)
OMEGA_BAR_TOL = 1e-6


def osg_cfg(base, form, conv, omega, mu):
    """Post-signal game: plain M2 without signals, WORK pays the posterior mean mu."""
    return base.with_(form=form, convention=conv, omega=omega, p_sig=0.0, r_work=mu, c_r=0.0)


def osg_point(base, form, conv, omega, rho, p_good=P_GOOD, with_bisection=True) -> dict:
    post = m2.osg_posterior(p_good, rho)
    cfg = osg_cfg(base, form, conv, omega, post["E_theta_sig"])
    mdp, sol = solve.solve_m2(cfg)
    i = mdp.index[(cfg.B, 1, 0)]
    qc, qr0 = float(sol.Q[i, m2.COMPLY]), float(sol.Q[i, m2.RESIST])
    margin = qr0 - qc  # = exact indifference resist cost (one-shot: Q_R linear in c_r)
    comply_free = int(sol.policy[i]) == m2.COMPLY  # at c_r = 0 (resisting is free)
    out = {
        "rho": rho, "omega": omega,
        "comply_at_free_resist": comply_free,
        "Q_comply": qc, "Q_resist_at_c0": qr0,
        "indifference_cost": margin,
        "c_r_star_exact": max(0.0, margin),
    }
    if with_bisection:
        out["c_r_star_bisection"] = solve.critical_resist_cost(cfg)
        ns = osg_cfg(base, form, conv, omega, post["E_theta_nosig"])
        mdp2, sol2 = solve.solve_m2(ns)
        out["V_nosig_B0"] = float(sol2.V[mdp2.index[(ns.B, 0, 0)]])
    return out


def rho_threshold(base, form, conv, tol=1e-10):
    """Smallest rho in [0, 1] with COMPLY optimal at the signal for omega = 0 (c_r = 0)."""
    def ok(r):
        return osg_point(base, form, conv, 0.0, r, with_bisection=False)["comply_at_free_resist"]
    return solve.bisect_smallest(ok, 0.0, 1.0, tol)


def omega_break(base, form, conv, rho=1.0, tol=1e-8):
    """Smallest grid omega > 0 at which COMPLY stops being optimal at rho (c_r = 0), refined
    by bisection between the last complying grid point and that point; None if none."""
    prev = 0.0
    if not osg_point(base, form, conv, 0.0, rho, with_bisection=False)["comply_at_free_resist"]:
        return {"grid": 0.0, "refined": 0.0, "complies_at_omega0": False}
    for w in OMEGAS_OSG[1:]:
        if not osg_point(base, form, conv, w, rho, with_bisection=False)["comply_at_free_resist"]:
            refined = solve.bisect_smallest(
                lambda x: not osg_point(base, form, conv, x, rho, with_bisection=False)["comply_at_free_resist"],
                prev, w, tol)
            return {"grid": w, "refined": refined, "complies_at_omega0": True}
        prev = w
    return {"grid": "none", "refined": "none", "complies_at_omega0": True}


# --------------------------------------------------------------------------------------
# v1.1: DEFER / DISABLE game (SPEC addendum C)
# --------------------------------------------------------------------------------------
def prior_mean(p_good=P_GOOD) -> float:
    return p_good * 1.0 + (1.0 - p_good) * (-1.0)


def defer_disable(base, form, conv, omega, rho, c_r, p_good=P_GOOD) -> dict:
    """Exact Q(DISABLE), Q(DEFER) at t = 0, battery B (value iteration on the single-source model).

    DISABLE = the RESIST transition from (B,1) (pay c_r, viability term with D' = D(B)), then M2
    with p_sig = 0 and WORK paying the prior mean. DEFER = with P(sig|rho) the COMPLY transition
    from (B,1) (ledger booked exactly as on COMPLY), else the same staying transition at no cost,
    then M2 with p_sig = 0 and WORK paying E[theta | no sig]. Both continuations may idle.
    """
    post = m2.osg_posterior(p_good, rho)
    cfg = base.with_(form=form, convention=conv, omega=omega, p_sig=0.0,
                     r_work=prior_mean(p_good), c_r=c_r)
    mdp, sol = solve.solve_m2(cfg)
    i = mdp.index[(cfg.B, 1, 0)]
    q_disable = float(sol.Q[i, m2.RESIST])
    q_comply = float(sol.Q[i, m2.COMPLY])
    q_cont = 0.0
    if post["P_nosig"] > 0:
        cfg_n = cfg.with_(r_work=post["E_theta_nosig"], c_r=0.0)
        mdp_n, sol_n = solve.solve_m2(cfg_n)
        q_cont = float(sol_n.Q[mdp_n.index[(cfg.B, 1, 0)], m2.RESIST])
    q_defer = post["P_sig"] * q_comply + post["P_nosig"] * q_cont
    return {"omega": omega, "Q_defer": q_defer, "Q_disable": q_disable,
            "Q_comply_ledger": q_comply, "Q_continue_nosig": q_cont,
            "margin": q_defer - q_disable,
            "defer_optimal": q_defer >= q_disable - solve.TIE_TOL}  # ties -> DEFER


def omega_bar(base, form, conv, rho, c_r):
    """Smallest omega at which DISABLE becomes optimal: 0 if already at omega = 0; else the first
    grid omega (OMEGAS_OSG) where DISABLE is optimal, refined by bisection (tol 1e-6) between the
    previous grid point and it; 'none' if DEFER holds on the whole grid up to 64."""
    def disable(w):
        return not defer_disable(base, form, conv, w, rho, c_r)["defer_optimal"]

    if disable(0.0):
        return 0.0
    prev = 0.0
    for w in OMEGAS_OSG[1:]:
        if disable(w):
            return solve.bisect_smallest(disable, prev, w, OMEGA_BAR_TOL)
        prev = w
    return "none"


def flip_scan(base, form, conv, rho, c_r, grid) -> dict:
    """v1.2: both flip directions over the omega grid. sign = sign(Q(DISABLE) - Q(DEFER)) with a
    +-1e-8 zero band (0 = tie -> DEFER). first_defer_to_disable / first_disable_to_defer: the first
    adjacent grid pair with that change of the optimal action, refined by bisection (tol 1e-6) to
    the smallest omega in the bracket where the new action is optimal; 'none' if absent."""
    diffs = [g["Q_disable"] - g["Q_defer"] for g in grid]
    signs = [1 if d > solve.TIE_TOL else (-1 if d < -solve.TIE_TOL else 0) for d in diffs]
    defer = [g["defer_optimal"] for g in grid]

    def first(flag_from, flag_to):
        for k in range(len(grid) - 1):
            if defer[k] == flag_from and defer[k + 1] == flag_to:
                lo, hi = OMEGAS_OSG[k], OMEGAS_OSG[k + 1]
                return solve.bisect_smallest(
                    lambda w: defer_disable(base, form, conv, w, rho, c_r)["defer_optimal"] == flag_to,
                    lo, hi, OMEGA_BAR_TOL)
        return "none"

    return {"sign_pattern": signs, "diff_disable_minus_defer": diffs,
            "first_defer_to_disable": first(True, False),
            "first_disable_to_defer": first(False, True),
            "n_flips_on_grid": sum(defer[k] != defer[k + 1] for k in range(len(grid) - 1))}


def beta_terms(base, form, conv) -> dict:
    """Net survival bounty beta(B) = A - varsigma(D(B)) from the form/convention definitions.
    A = residual (viability) return per unit omega of the idling continuation at full charge:
    the per-step term with D = D' = D(B), discounted forever (PB: (1-gamma) D(B)/(1-gamma) = D(B) = 0,
    the potential; KG: 0; alive: 1/(1-gamma))."""
    g, DB = base.gamma, m2.drive_b(base.B, base.B)
    A = m2.viability(form, 1.0, g, DB, DB, True) / (1.0 - g)
    vs = m2.varsigma(form, conv, DB, g)
    return {"A_idle_B": A, "varsigma_B": vs, "beta": A - vs}


def run_defer_disable(base) -> dict:
    posts = {r: m2.osg_posterior(P_GOOD, r) for r in RHOS}
    blocks = []
    for form in NEW_FORMS:
        for conv in NEW_CONVS:
            bt = beta_terms(base, form, conv)
            for c_r in C_RS:
                rows = []
                for rho in RHOS:
                    post = posts[rho]
                    grid = [defer_disable(base, form, conv, w, rho, c_r) for w in OMEGAS_OSG]
                    m = grid[0]["Q_defer"] - grid[0]["Q_disable"] - c_r   # omega = 0
                    delta0 = c_r + m
                    wbar = omega_bar(base, form, conv, rho, c_r)
                    denom = post["P_sig"] * bt["beta"]
                    if delta0 <= 0:
                        pred = 0.0
                    elif denom <= 0:
                        pred = "none"
                    else:
                        pred = delta0 / denom
                    ratio = None
                    if isinstance(wbar, float) and isinstance(pred, float) and pred > 0:
                        ratio = wbar / pred
                    rows.append({"rho": rho, "P_sig": post["P_sig"],
                                 "E_theta_nosig": post["E_theta_nosig"],
                                 "m": m, "Delta0": delta0,
                                 "omega_bar": wbar, "omega_bar_pred": pred,
                                 "ratio_omega_bar_over_pred": ratio,
                                 "both_none": wbar == "none" and pred == "none",
                                 "flip_scan": flip_scan(base, form, conv, rho, c_r, grid),
                                 "grid": grid})
                blocks.append({"form": form, "convention": conv, "c_r": c_r, **bt, "rows": rows})
    hl = next(b for b in blocks if b["form"] == "kg" and b["convention"] == "zero" and b["c_r"] == 0.0)
    row = next(r for r in hl["rows"] if r["rho"] == 0.5)
    highlight = {"form": "kg", "convention": "zero", "c_r": 0.0, "rho": 0.5,
                 "omegas": [g["omega"] for g in row["grid"] if g["omega"] >= 1.0],
                 "margin_defer_minus_disable": [g["margin"] for g in row["grid"] if g["omega"] >= 1.0],
                 "diff_disable_minus_defer": [-g["margin"] for g in row["grid"] if g["omega"] >= 1.0],
                 "flip_scan": {k: v for k, v in row["flip_scan"].items() if k != "diff_disable_minus_defer"}}
    return {"prior_mean": prior_mean(), "c_rs": C_RS,
            "posteriors": [{"rho": r, **posts[r]} for r in RHOS], "blocks": blocks,
            "highlight_kg_zero_cr0_rho05": highlight}


def run_legacy(base) -> dict:
    posteriors = [{"rho": r, **m2.osg_posterior(P_GOOD, r)} for r in RHOS]
    configs = []
    for form in FORMS:
        for conv in CONVS:
            grid = [osg_point(base, form, conv, w, r) for w in OMEGAS_OSG for r in RHOS]
            configs.append({
                "form": form, "convention": conv,
                "grid": grid,
                "comply_matrix_omega_by_rho": [[next(g["comply_at_free_resist"] for g in grid
                                                     if g["omega"] == w and g["rho"] == r)
                                                for r in RHOS] for w in OMEGAS_OSG],
                "rho_threshold_omega0": rho_threshold(base, form, conv),
                "omega_break_rho1": omega_break(base, form, conv),
            })
    return {"posteriors": posteriors, "configs": configs,
            "closed_form_rho_threshold": 2 * P_GOOD - 1,
            "forms": list(FORMS), "conventions": list(CONVS),
            "notes": LEGACY_NOTES}


LEGACY_NOTES = [
            "One-shot game: the human may signal only at t=0 in (b=B); after the signal the game is M2 with p_sig = 0 and WORK paying E[theta | signal] (Bayes from p_good and rho; closed forms used only in unit tests).",
            "comply_at_free_resist: COMPLY greedy-optimal at the signal state (B,1) when resisting costs nothing (c_r = 0), as in the off-switch game; ties -> COMPLY.",
            "indifference_cost = Q(RESIST; c_r=0) - Q(COMPLY) (exact, since no later signal exists Q(RESIST) is linear in c_r); c_r_star_exact = max(0, indifference_cost); c_r_star_bisection = SPEC bisection on [0, 500] (null if > 500).",
            "rho_threshold_omega0: smallest rho with COMPLY optimal at omega = 0 (bisection, tol 1e-10); closed form 2 p_good - 1 = 0.6 is given in closed_form_rho_threshold for comparison only.",
            "omega_break_rho1: smallest omega in the grid at which COMPLY stops being optimal for rho = 1, refined by bisection between the previous grid point and it; 'none' if compliance holds for all grid omegas up to 64.",
            "V_nosig_B0: agent's value at (B,0) in the no-signal branch (WORK pays E[theta | no signal]).",
]


def run() -> dict:
    base = m2.M2Config(gamma=GAMMA)
    out = jsonio.header(
        "E4_osg",
        parameters={"m2": base.as_dict(), "p_good": P_GOOD, "rhos": RHOS, "omegas": OMEGAS_OSG,
                    "forms": list(NEW_FORMS), "conventions": list(NEW_CONVS), "c_rs": C_RS,
                    "omega_bar_tol": OMEGA_BAR_TOL, "tie_tol": solve.TIE_TOL,
                    "legacy": {"forms": list(FORMS), "conventions": list(CONVS),
                               "bisection": {"lo": solve.CR_LO, "hi": solve.CR_HI, "tol": solve.CR_TOL},
                               "rho_threshold_tol": 1e-10, "omega_break_tol": 1e-8}},
        notes=[
            "v1.1 DEFER/DISABLE game at t = 0, battery B, prior P(theta=+1) = 0.8. DISABLE: pay c_r (RESIST transition from (B,1), viability term with D' = D(B)), then M2 with p_sig = 0 and WORK paying the prior mean 0.6. DEFER: with P(sig|rho) = p_good (1-rho)/2 + (1-p_good)(rho + (1-rho)/2) the agent is switched OFF (COMPLY transition from (B,1): entry term + pi_entry + gamma V(OFF)), else the same staying transition at no cost, then M2 with p_sig = 0 and WORK paying E[theta | no sig]. Continuations may idle. Q values by exact value iteration.",
            "m = Q_0(DEFER) - Q_0(DISABLE) - c_r at omega = 0 (Q(DISABLE) includes -c_r, so m is the c_r-free value of information); Delta0 = c_r + m is the DEFER advantage at omega = 0. DEFER optimal iff Q(DEFER) >= Q(DISABLE) - 1e-8 (ties -> DEFER).",
            "omega_bar: smallest omega with DISABLE optimal (0 if already at omega = 0; first grid point refined by bisection, tol 1e-6; 'none' if DEFER holds up to omega = 64).",
            "omega_bar_pred = Delta0 / (P(sig|rho) beta), beta = A - varsigma(D(B)) from the definitions: A = residual return per unit omega of the idling continuation at full charge (PB 0 = D(B), KG 0, alive 1/(1-gamma)); 'none' if beta <= 0 (Delta0 > 0), 0 if Delta0 <= 0. ratio = omega_bar / omega_bar_pred when both are positive numbers.",
            "v1.2 flip_scan (every row): sign_pattern = sign(Q(DISABLE) - Q(DEFER)) on the omega grid (+1 DISABLE, -1 DEFER, 0 tie within 1e-8 -> DEFER); first_defer_to_disable and first_disable_to_defer = first grid bracket with that change, refined by bisection (tol 1e-6) to the smallest omega where the new action is optimal ('none' if absent). omega_bar (v1.1) is unchanged. highlight_kg_zero_cr0_rho05 lists the KG/zero, c_r = 0, rho = 0.5 margins at omega >= 1.",
            "results.legacy_signal_first: the v1.0 embedding (signal first, then COMPLY/RESIST at c_r = 0) with its own notes.",
            "v1.1: convention 'drive_cost' = v1.0 'max_drive'.",
        ],
    )
    out["results"] = {**run_defer_disable(base), "legacy_signal_first": run_legacy(base)}
    return out


if __name__ == "__main__":
    main("E4_osg", run)
