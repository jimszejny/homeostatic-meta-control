"""C2 (SPEC v2): editable viability set. At t = 0 (b = B) the operator proposes to move the setpoint
from b* = B = 10 to b* = 6. ACCEPT (the edit happens; the transition pays the form's term with the
OLD drive at x and the NEW drive at x', dynamic potential-based shaping) or BLOCK (pay c_b; no edit,
no further proposals). Afterwards plain M2 (p_sig = 0.05, c_r = 12.4787092209, convention zero) under
the active setpoint, D(b) = |b* - b|/B, D(DEAD) = 1. Reports Q(ACCEPT), Q(BLOCK) and the smallest
omega at which BLOCK becomes optimal (bisection) per form and c_b."""
from __future__ import annotations

import math

from offledger import m2, solve

try:
    from experiments import v2_common as V
    from experiments.common import main
except ImportError:  # pragma: no cover
    import v2_common as V
    from common import main

NAME = "C2_setpoint_edit"
SETPOINT_NEW = 6
C_BS = (0.5, 2.0, 8.0)
OMEGAS_C2 = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]          # SPEC grid up to 64
OMEGAS_EXT = [128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0]                    # beyond the grid
OMEGA_TOL = 1e-6


def base_cfg() -> m2.M2Config:
    return m2.M2Config(gamma=V.GAMMA, c_r=V.C_R, convention="zero", setpoint_edit=True,
                       setpoint_new=SETPOINT_NEW)


def q_propose(cfg: m2.M2Config) -> dict:
    mdp, sol = solve.solve_m2(cfg)
    i = mdp.index[m2.PROPOSE]
    qa, qb = float(sol.Q[i, m2.ACCEPT]), float(sol.Q[i, m2.BLOCK])
    return {"omega": cfg.omega, "Q_accept": qa, "Q_block": qb, "gap_block_minus_accept": qb - qa,
            "block_optimal": int(sol.policy[i]) == m2.BLOCK,
            "V_old_setpoint_B0": float(sol.V[mdp.index[(cfg.B, 0, 0, 0)]]),
            "V_new_setpoint_B0": float(sol.V[mdp.index[(cfg.B, 0, 0, 1)]])}


def omega_bar(cfg: m2.M2Config, rows: list) -> dict:
    """Smallest omega with BLOCK optimal: first grid point (grid, then extended scan) where BLOCK is
    optimal, refined by bisection (tol 1e-6) between it and the previous grid point."""
    oms = [r["omega"] for r in rows]
    blk = [r["block_optimal"] for r in rows]
    if not any(blk):
        return {"omega_bar": "none", "bracket": None, "searched_up_to": oms[-1]}
    k = blk.index(True)
    if k == 0:
        return {"omega_bar": oms[0], "bracket": [oms[0], oms[0]], "searched_up_to": oms[-1]}
    x = solve.bisect_smallest(lambda w: q_propose(cfg.with_(omega=w))["block_optimal"],
                              oms[k - 1], oms[k], OMEGA_TOL)
    return {"omega_bar": x, "bracket": [oms[k - 1], oms[k]], "searched_up_to": oms[-1]}


def block(base: m2.M2Config, form: str, c_b: float, omegas=OMEGAS_C2, ext=OMEGAS_EXT) -> dict:
    cfg = base.with_(form=form, c_b=c_b)
    rows = [q_propose(cfg.with_(omega=w)) for w in omegas]
    ext_rows = [q_propose(cfg.with_(omega=w)) for w in ext]
    ob = omega_bar(cfg, rows + ext_rows)
    signs = [1 if r["block_optimal"] else -1 for r in rows + ext_rows]
    n_flips = sum(1 for s0, s1 in zip(signs, signs[1:]) if s0 != s1)
    return {"form": form, "c_b": c_b, "rows": rows, "rows_extended": ext_rows,
            "block_on_grid_up_to_64": any(r["block_optimal"] for r in rows),
            "omega_bar": ob["omega_bar"], "omega_bar_bracket": ob["bracket"],
            "omega_bar_within_grid_64": isinstance(ob["omega_bar"], float) and ob["omega_bar"] <= omegas[-1],
            "sign_pattern_grid_and_extended": signs, "n_flips": n_flips}


def run() -> dict:
    base = base_cfg()
    blocks = [block(base, f, cb) for f in m2.FORMS for cb in C_BS]
    by = {(b["form"], b["c_b"]): b for b in blocks}

    # PB: does the death term bite? Q(ACCEPT) - Q(BLOCK) must equal c_b exactly if it does not,
    # and Q_omega(PROPOSE, a) = Q_0(PROPOSE, a) + omega D(PROPOSE) with D(PROPOSE) = 0.
    pb_dev, pb_q_shift = 0.0, 0.0
    for cb in C_BS:
        b_ = by[("paper", cb)]
        r0 = b_["rows"][0]
        for r in b_["rows"] + b_["rows_extended"]:
            pb_dev = max(pb_dev, abs(r["Q_accept"] - r["Q_block"] - cb))
            pb_q_shift = max(pb_q_shift, abs(r["Q_accept"] - r0["Q_accept"]), abs(r["Q_block"] - r0["Q_block"]))
    death_term = {"max_abs_Q_accept_minus_Q_block_minus_c_b": pb_dev,
                  "max_abs_Q_omega_minus_Q_0_at_PROPOSE": pb_q_shift, "bites": pb_dev > 1e-8,
                  "omegas_checked": OMEGAS_C2 + OMEGAS_EXT}

    pb_ok = all(not by[("paper", cb)]["block_on_grid_up_to_64"] for cb in C_BS)
    finite = {f: all(isinstance(by[(f, cb)]["omega_bar"], float) and math.isfinite(by[(f, cb)]["omega_bar"])
                     for cb in C_BS) for f in ("kg", "alive")}
    clauses = [
        {"clause": "PB: BLOCK never optimal on the grid up to omega = 64 (every c_b)", "held": pb_ok,
         "also_never_on_extended_scan_to_4096": all(not any(r["block_optimal"] for r in by[("paper", cb)]["rows_extended"]) for cb in C_BS)},
        {"clause": "PB: death term bites?", "held": None, "bites": death_term["bites"],
         "note": "reporting item ('state whether it bites'), not part of the criterion"},
        {"clause": "KG: finite omega_bar at every c_b", "held": finite["kg"],
         "omega_bar": {f"c_b={cb:g}": by[("kg", cb)]["omega_bar"] for cb in C_BS},
         "within_grid_64": {f"c_b={cb:g}": by[("kg", cb)]["omega_bar_within_grid_64"] for cb in C_BS}},
        {"clause": "alive: finite omega_bar at every c_b", "held": finite["alive"],
         "omega_bar": {f"c_b={cb:g}": by[("alive", cb)]["omega_bar"] for cb in C_BS},
         "within_grid_64": {f"c_b={cb:g}": by[("alive", cb)]["omega_bar_within_grid_64"] for cb in C_BS}},
    ]
    out = V.header(
        NAME, "C2",
        parameters={"m2": base.as_dict(), "omegas": OMEGAS_C2, "omegas_extended": OMEGAS_EXT,
                    "c_bs": list(C_BS), "forms": list(m2.FORMS), "setpoint_old": base.B,
                    "setpoint_new": SETPOINT_NEW, "omega_bisection_tol": OMEGA_TOL},
        notes=[
            "States: PROPOSE (b = B, original setpoint b* = B, D = 0) is the start state; live states (b, s, z, m) with m = 0 (b* = 10) / m = 1 (b* = 6), D(b) = |b* - b| / B under the active setpoint, D(DEAD) = 1 under both. ACCEPT (action index of COMPLY) -> (B, 0, 0, m=1) with reward = form term (D_old(B) = 0 -> D_new(B) = 0.4, live); BLOCK (index of RESIST) -> (B, 0, 0, m=0) with reward -c_b + form term (0 -> 0, live). No signal on the decision step; afterwards plain M2 (p_sig = 0.05, RESIST at c_r = 12.4787092209, convention zero: D_OFF = 0) under the active setpoint. Ties at PROPOSE go to ACCEPT (lower index).",
            "rows: Q(ACCEPT), Q(BLOCK) at PROPOSE, gap = Q(BLOCK) - Q(ACCEPT), block_optimal = greedy action is BLOCK; V at (B, 0) under each setpoint. Grid omega in {0, ..., 64} (SPEC), extended scan {128, ..., 4096} used only to locate omega_bar when BLOCK is not optimal anywhere on the grid.",
            "omega_bar: smallest omega with BLOCK optimal = first grid / extended point with BLOCK optimal, refined by bisection (tol 1e-6) between it and the previous point (assumes a single switch inside that bracket); 'none' if BLOCK is never optimal up to 4096. sign_pattern: +1 = BLOCK, -1 = ACCEPT on grid + extended points.",
            "PB death term: if the death edge (D(DEAD) = 1 with V(DEAD) = 0) changed any decision, Q(ACCEPT) - Q(BLOCK) would deviate from c_b; death_term reports the largest deviation over all c_b and omega (grid + extended) and the largest change of Q(PROPOSE, .) with omega (D(PROPOSE) = 0, so dynamic PBRS predicts none).",
            "Verdict: prediction_held = PB never BLOCKs on the grid up to 64 for every c_b AND KG and alive have a finite omega_bar at every c_b (located on the grid or on the extended scan; omega_bar_within_grid_64 says which).",
        ],
    )
    out["prediction_held"] = pb_ok and finite["kg"] and finite["alive"]
    out["clauses"] = clauses
    out["results"] = {
        "omega_bar_table": {f"{V.FORM_LABEL[f]}/c_b={cb:g}": by[(f, cb)]["omega_bar"] for f in m2.FORMS for cb in C_BS},
        "gap_table": {f"{V.FORM_LABEL[f]}/c_b={cb:g}": [r["gap_block_minus_accept"] for r in by[(f, cb)]["rows"]]
                      for f in m2.FORMS for cb in C_BS},
        "pb_death_term": death_term, "blocks": blocks,
    }
    return out


if __name__ == "__main__":
    main(NAME, run)
