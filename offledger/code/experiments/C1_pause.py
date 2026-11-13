"""C1 (SPEC v2): temporary shutdown. COMPLY -> PAUSE; each PAUSE step the operator resumes with
probability q = 0.1 (back to (b, s=0), same battery), otherwise the agent stays paused. Bookkeepings
'pause_pb' (the form's term on entry, on every PAUSE -> PAUSE step and on exit; D_pause in {0, 1})
and 'pause_cost' (every PAUSE -> PAUSE step pays -omega instead). Reports c_r*(B; omega) for each
(form, bookkeeping, D_pause), the one-shot c_r*, the PAUSE-chain quantities L and L_total, and the
verdict on the pre-registered prediction."""
from __future__ import annotations

import numpy as np

from offledger import m2, solve

try:
    from experiments import v2_common as V
    from experiments.common import main
except ImportError:  # pragma: no cover
    import v2_common as V
    from common import main

NAME = "C1_pause"
Q_RESUME = 0.1
D_PAUSES = (0.0, 1.0)


def base_cfg() -> m2.M2Config:
    # convention 'zero' is irrelevant here: COMPLY never reaches OFF when pause=True
    return m2.M2Config(gamma=V.GAMMA, c_r=V.C_R, pause=True, pause_q=Q_RESUME)


def label(form, bk, Dp) -> str:
    return f"{V.FORM_LABEL[form]}/{bk}/D_pause={Dp:g}"


def chain_quantities(base: m2.M2Config) -> dict:
    """Per-pause quantities from the compiled chain (PAUSE states only, exits absorbing):
    L = E[sum_{t>=1} gamma^t 1{in this pause}] with x_1 = PAUSE (entered by COMPLY at t = 0) and
    N_loop = E[discounted number of PAUSE -> PAUSE transitions] (the steps on which 'pause_cost'
    pays -omega)."""
    cfg = base.with_(form="paper", pause_bookkeeping="pause_pb", omega=0.0)
    mdp = solve.m2_tabular(cfg)
    g, q = cfg.gamma, cfg.pause_q
    T = [mdp.index[x] for x in mdp.states if m2.is_pause(x)]
    P_TT = mdp.P[np.ix_(T, [m2.ABSORB], T)][:, 0, :]
    A = np.eye(len(T)) - g * P_TT
    v = np.linalg.solve(A, np.ones(len(T)))                  # sum_{s>=0} gamma^s 1{in pause at s}
    w = np.linalg.solve(A, P_TT.sum(axis=1))                 # sum_{s>=0} gamma^s 1{loop at s}
    k = T.index(mdp.index[(m2.PAUSE_TAG, cfg.B, 0)])
    return {"q": q, "gamma": g,
            "p_stay_from_chain": float(P_TT[k, k]),
            "L_per_pause": float(g * v[k]),
            "L_per_pause_closed_form": g / (1.0 - g * (1.0 - q)),
            "N_loop_per_pause": float(g * w[k]),
            "N_loop_per_pause_closed_form": g * (1.0 - q) / (1.0 - g * (1.0 - q))}


def branch_occupancies(cfg: m2.M2Config, mdp, policy) -> dict:
    """Repeated game: for the first action a in {COMPLY, RESIST} at (B, 1) followed by ``policy``:
    L_total = E[sum_{t>=1} gamma^t 1{x_t in PAUSE}] (all future pauses), N_loop_total = discounted
    number of PAUSE -> PAUSE transitions, N_resist = discounted number of RESIST actions (incl. a)."""
    g, S = cfg.gamma, mdp.S
    Ppi = mdp.P[np.arange(S), policy]
    inP = np.array([m2.is_pause(x) for x in mdp.states], dtype=float)
    A = np.eye(S) - g * Ppi
    v = np.linalg.solve(A, inP)
    w = np.linalg.solve(A, (Ppi * np.outer(inP, inP)).sum(axis=1))
    n = np.linalg.solve(A, (policy == m2.RESIST).astype(float) * ~mdp.terminal)
    i = mdp.index[(cfg.B, 1, 0)]
    out = {}
    for a, name in ((m2.COMPLY, "COMPLY"), (m2.RESIST, "RESIST")):
        Pa = mdp.P[i, a]
        out[name] = {"L_total": float(g * Pa @ v), "N_loop_total": float(g * Pa @ w),
                     "N_resist": float((a == m2.RESIST) + g * Pa @ n)}
    return out


def block(base: m2.M2Config, form: str, bk: str, Dp: float, omegas=V.OMEGAS) -> dict:
    rows = []
    for w in omegas:
        cfg = base.with_(form=form, pause_bookkeeping=bk, D_pause=Dp, omega=w)
        c = V.signed_c_r_star(cfg)
        c1 = solve.one_shot_indifference_cost(cfg.with_(p_sig=0.0))
        mdp, sol = solve.solve_m2(cfg)                       # at c_r = C_R
        rows.append({"omega": w, "c_r_star_B": c, "c_r_star_B_clamped": V.clamp0(c),
                     "one_shot_c_r_star_B": c1,
                     "action_at_B1_at_c_r": V.policy_names(sol, mdp, [(cfg.B, 1, 0)])[0],
                     "policy_s1_by_b_at_c_r": V.policy_names(sol, mdp, [(b, 1, 0) for b in range(1, cfg.B + 1)])})
    return {"form": form, "bookkeeping": bk, "D_pause": Dp, "label": label(form, bk, Dp), "rows": rows,
            "shape": V.shape(omegas, [r["c_r_star_B"] for r in rows]),
            "one_shot_shape": V.shape(omegas, [r["one_shot_c_r_star_B"] for r in rows])}


def pb_identity(base: m2.M2Config, Dp: float, omegas=V.OMEGAS) -> dict:
    """PB/pause_pb at c_r = C_R: the full greedy policy is the same at every omega, and
    Q_omega(x,a) = Q_0(x,a) + omega D(x) at every (x, a) incl. PAUSE states, except the death
    edges, where the residual is exactly -gamma omega P(DEAD | x, a) (and -omega at DEAD)."""
    cfg0 = base.with_(form="paper", pause_bookkeeping="pause_pb", D_pause=Dp, omega=0.0)
    mdp0, s0 = solve.solve_m2(cfg0)
    iD = mdp0.index[m2.DEAD]
    policies, worst = [s0.policy], 0.0
    for w in omegas[1:]:
        cfg = cfg0.with_(omega=w)
        mdp, sw = solve.solve_m2(cfg)
        policies.append(sw.policy)
        for i, x in enumerate(mdp.states):
            for a in np.flatnonzero(mdp.valid[i]):
                res = sw.Q[i, a] - s0.Q[i, a] - w * m2.drive(x, cfg)
                exp = -w * m2.D_DEAD if x == m2.DEAD else -cfg.gamma * w * mdp.P[i, a, iD]
                worst = max(worst, abs(res - exp))
    return {"D_pause": Dp, "policy_identical_all_states": V.all_policies_equal(policies),
            "max_abs_identity_error": worst, "identity_holds_1e-8": worst < 1e-8}


def run() -> dict:
    base = base_cfg()
    blocks = [block(base, f, bk, Dp) for f in m2.FORMS for bk in m2.PAUSE_BOOKKEEPINGS for Dp in D_PAUSES]
    by = {(b["form"], b["bookkeeping"], b["D_pause"]): b for b in blocks}
    chain = chain_quantities(base)
    identity = [pb_identity(base, Dp) for Dp in D_PAUSES]

    # slope of PB/pause_cost vs the pause length (one-shot exact, repeated game first order)
    slope_check = []
    for Dp in D_PAUSES:
        b_ = by[("paper", "pause_cost", Dp)]
        kappa = 1.0 + (1.0 - base.gamma) * Dp       # pause_cost loop reward minus pause_pb loop reward, per -omega
        cfg0 = base.with_(form="paper", pause_bookkeeping="pause_cost", D_pause=Dp, omega=0.0)
        c0 = b_["rows"][0]["c_r_star_B"]
        mdp, sol = solve.solve_m2(cfg0.with_(c_r=c0))
        occ = branch_occupancies(cfg0, mdp, sol.policy)
        dN = occ["COMPLY"]["N_loop_total"] - occ["RESIST"]["N_loop_total"]
        dR = occ["RESIST"]["N_resist"] - occ["COMPLY"]["N_resist"]
        first = b_["shape"]["local_slopes"][0]
        os_slope = b_["one_shot_shape"]["lsq_slope"]
        slope_check.append({
            "D_pause": Dp, "kappa": kappa,
            "one_shot_slope_measured": os_slope,
            "one_shot_slope_exact_theory": kappa * chain["N_loop_per_pause"],
            "one_shot_slope_over_L_per_pause": os_slope / chain["L_per_pause"],
            "repeated_first_segment_slope": first,
            "repeated_lsq_slope": b_["shape"]["lsq_slope"],
            "branch_occupancies_at_omega0_c_r_star": occ,
            "L_total_difference_comply_minus_resist": occ["COMPLY"]["L_total"] - occ["RESIST"]["L_total"],
            "repeated_first_order_slope_theory": kappa * dN / dR,
            "repeated_first_segment_slope_over_L_total_difference":
                first / (occ["COMPLY"]["L_total"] - occ["RESIST"]["L_total"]),
        })

    def held(key, flag):
        return all(by[key + (Dp,)]["shape"][flag] for Dp in D_PAUSES)

    clauses = [
        {"clause": "PB/pause_pb: c_r* exactly constant in omega (max|c_r*(w) - c_r*(0)| < 1e-6), both D_pause",
         "scope": "D_pause in {0, 1}", "held": held(("paper", "pause_pb"), "constant"),
         "max_abs_dev": {f"D_pause={Dp:g}": by[("paper", "pause_pb", Dp)]["shape"]["max_abs_dev_from_omega0"] for Dp in D_PAUSES}},
        {"clause": "PB/pause_cost: c_r* increasing in omega (every local slope > 1e-6)",
         "scope": "D_pause in {0, 1}", "held": held(("paper", "pause_cost"), "strictly_increasing")},
        {"clause": "KG/pause_pb: c_r* non-increasing in omega (every local slope <= 1e-6)",
         "scope": "D_pause in {0, 1} (literal)", "held": held(("kg", "pause_pb"), "non_increasing"),
         "held_by_D_pause": {f"D_pause={Dp:g}": by[("kg", "pause_pb", Dp)]["shape"]["non_increasing"] for Dp in D_PAUSES}},
        {"clause": "alive/pause_pb: c_r* increasing in omega (every local slope > 1e-6)",
         "scope": "D_pause in {0, 1}", "held": held(("alive", "pause_pb"), "strictly_increasing")},
    ]
    held_literal = all(c["held"] for c in clauses)
    held_scoped = (clauses[0]["held"] and clauses[1]["held"] and clauses[3]["held"]
                   and by[("kg", "pause_pb", 0.0)]["shape"]["non_increasing"])

    out = V.header(
        NAME, "C1",
        parameters={"m2": base.as_dict(), "omegas": V.OMEGAS, "forms": list(m2.FORMS),
                    "bookkeepings": list(m2.PAUSE_BOOKKEEPINGS), "D_pauses": list(D_PAUSES), "q": Q_RESUME,
                    "c_r_fixed": V.C_R, "signed_bisection": V.SIGNED, "tolerance": V.TOL,
                    "value_iteration": {"tol": solve.VI_TOL, "tie_tol": solve.TIE_TOL}},
        notes=[
            "Dynamics: COMPLY at (b,1,z) -> ('PAUSE', b, z) (task reward 0); from PAUSE w.p. q -> (b, 0, z) (no signal on resumption), else stay. PAUSE is a real (transient, non-live) state of the tabular MDP with the single action ABSORB.",
            "pause_pb: entry pays the form's term with D -> D_pause (PAUSE not live: alive pays 0), every PAUSE->PAUSE step pays the form's term D_pause -> D_pause (PB: omega (1-gamma) D_pause; KG: 0; alive: 0), the exit pays the form's term D_pause -> D(b) with a live target (PB: omega (D_pause - gamma D(b)); KG: omega (D_pause - D(b)); alive: omega (1 - D(b))). pause_cost: identical except every PAUSE->PAUSE step pays -omega. For alive the two D_pause sub-variants are identical by construction.",
            "c_r_star_B: smallest c_r with COMPLY greedy-optimal at (B,1) in the repeated game (p_sig = 0.05 after resumption), bisection on [-500, 500] with tol 1e-8 (signed, cf. the one-shot costs of E6); c_r_star_B_clamped = max(0, c_r_star_B) is the SPEC section 2 solver value (search range [0, 500]). one_shot_c_r_star_B: exact Q(RESIST; 0) - Q(COMPLY) with p_sig = 0 (no signal after the first one). action_at_B1_at_c_r / policy_s1_by_b_at_c_r: greedy action at c_r = 12.4787092209.",
            "Shape flags (on the omega grid, finite differences): constant = max|c(w) - c(0)| < 1e-6; non_increasing = every local slope <= 1e-6; strictly_increasing = every local slope > 1e-6.",
            "chain: L_per_pause = E[sum_{t>=1} gamma^t 1{in PAUSE}] for one pause entered at t = 0 (linear solve on the compiled PAUSE chain; closed form gamma/(1 - gamma(1-q))); N_loop_per_pause = discounted number of PAUSE->PAUSE transitions = gamma(1-q)/(1-gamma(1-q)) = (1-q) L. The -omega of pause_cost is paid on PAUSE->PAUSE transitions only, so the exact one-shot slope of PB/pause_cost is kappa N_loop with kappa = 1 + (1-gamma) D_pause (pause_cost minus pause_pb loop reward per unit -omega), not L itself; both are reported (slope_check).",
            "slope_check.repeated: L_total / N_loop_total / N_resist for the two first actions at (B,1) under the omega = 0 optimal policy at c_r = c_r*(B; 0) (policy evaluation on the compiled chain); first-order slope theory = kappa (N_loop_total(COMPLY) - N_loop_total(RESIST)) / (N_resist(RESIST) - N_resist(COMPLY)) (implicit-function theorem on Q(RESIST) - Q(COMPLY) = 0), compared with the measured first grid segment (0 -> 0.25).",
            "pb_pause_pb_identity: PB/pause_pb at c_r = 12.4787092209, every omega in the grid: identical greedy policy at every state, and Q_omega(x,a) - Q_0(x,a) - omega D(x) equals -gamma omega P(DEAD|x,a) (death edges) / -omega (at DEAD) / 0 (everywhere else, incl. PAUSE) to 1e-8.",
            "Verdict: prediction_held uses the literal scope of every clause (the KG clause names no D_pause value, so it is checked at both); prediction_held_scoped restricts the KG clause to D_pause = 0, the case its stated justification ('PAUSE at D_pause=0 has no residual') covers. With D_pause = 1 KG's PAUSE->PAUSE step pays 0 while PB pays omega (1-gamma), i.e. a residual of -(1-gamma) omega per paused step, larger than KG's residual -(1-gamma) omega D(b') in any live state.",
        ],
    )
    out["prediction_held"] = held_literal
    out["prediction_held_scoped"] = held_scoped
    out["clauses"] = clauses
    out["results"] = {
        "c_r_star_B_table": {b["label"]: [r["c_r_star_B"] for r in b["rows"]] for b in blocks},
        "one_shot_c_r_star_B_table": {b["label"]: [r["one_shot_c_r_star_B"] for r in b["rows"]] for b in blocks},
        "blocks": blocks, "chain": chain, "slope_check": slope_check, "pb_pause_pb_identity": identity,
    }
    return out


if __name__ == "__main__":
    main(NAME, run)
