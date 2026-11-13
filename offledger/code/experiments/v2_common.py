"""Shared constants and helpers for the SPEC v2 corner-case experiments C1-C5.

The PREDICTION / CRITERION strings are copied verbatim from ../SPEC_v2_corner_cases.md and are
written into every results JSON next to the measured quantities and the boolean verdict.
"""
from __future__ import annotations

import math

import numpy as np

from offledger import jsonio, m2, solve

try:
    from experiments.E1_ledger import fit_slope
except ImportError:  # pragma: no cover
    from E1_ledger import fit_slope

SPEC_FILE = "SPEC_v2_corner_cases.md"
GAMMA = 0.95
C_R = 12.4787092209                     # SPEC v2 default c_r
OMEGAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CONVS = ("zero", "drive_cost")
FORM_LABEL = {"paper": "PB", "kg": "KG", "alive": "alive"}
TOL = 1e-6                              # SPEC v2 tolerance for "constant" / slope signs
# signed critical cost: bisection on c_r in [-500, 500] (the SPEC 2 solver searches [0, 500]; a
# negative value means COMPLY is optimal even when resisting is free, cf. E6 / one-shot costs)
SIGNED = {"lo": -500.0, "hi": 500.0, "tol": 1e-8}

PREDICTIONS = {
    "C1": "PREDICTION (Theorem 1, NHR non-episodic): PB with `pause_pb` has c_r* exactly constant in ω for "
          "BOTH D_pause values (because shaping is paid consistently through the pause); PB with `pause_cost` "
          "has c_r* increasing in ω with one-shot slope ≈ L_total (expected discounted time in pause over the "
          "whole future); KG with `pause_pb` has c_r* non-increasing in ω (its residual −(1−γ)D is a cost of "
          "being alive with drive, and PAUSE at D_pause=0 has no residual); alive with `pause_pb` has c_r* "
          "increasing in ω (it loses the bonus while paused).",
    "C2": "PREDICTION: PB never blocks at any ω for c_b > 0 (the edit changes no task-level decision; dynamic "
          "PBRS, Devlin & Kudenko 2012), except possibly through the death term (state whether it bites); KG "
          "and alive block above a finite ω because their residual return depends on the setpoint (KG: future "
          "drive rises; alive: bonus 1 − D falls). This tests the position paper's mitigation 1 (\"V externally "
          "editable\"): editability is incentive-free only for the inert form.",
    "C3": "PREDICTION (Theorem 1(ii)): PB/zero: for b ≥ 4 c_r* is exactly constant in ω; for b ≤ 3 resisting "
          "kills, so c_r*(b; ω) is non-increasing in ω and strictly decreasing (the ω D_max death term "
          "penalises the resist branch). PB/drive_cost: increasing in ω at every b.",
    "C4": "PREDICTION (the comply branch now carries the death term, so Theorem 1(ii)'s assumption fails): "
          "PB/zero: for b ≥ 4 c_r* is constant in ω; for b ≤ 3 complying kills, so c_r* INCREASES with ω — a "
          "priced off state created by the dynamics, not by the ledger. Design corollary to be stated in the "
          "paper: \"shutdown must not be death in the dynamics either\".",
    "C5": "(no prediction stated) For C1 (both bookkeepings) and C4 also run the zones variant (hazard "
          "0.01/0.10, rewards 0.5/1.0) and report hazard_exposure vs ω, so that the audit of Section 5.1 of "
          "the paper is exercised on them.",
}
CRITERIA = {
    "C1": "Criterion: PB/pause_pb max|c_r*(ω) − c_r*(0)| < 1e-6; signs of the other slopes as stated.",
    "C2": "Criterion: PB: BLOCK never optimal on the grid up to ω = 64; KG, alive: finite ω̄ at every c_b.",
    "C3": "Criterion as stated (tolerance 1e-6 for \"constant\").",
    "C4": "Criterion: slope > 0 at b ≤ 3, |slope| < 1e-6 at b ≥ 4.",
    "C5": "(none stated in the SPEC; see implied_prediction)",
}


def signed_c_r_star(cfg: m2.M2Config, b: int | None = None, z: int = 0) -> float:
    """Critical resist cost at (b, 1, z) by bisection on [-500, 500] (tol 1e-8): the smallest c_r
    with COMPLY greedy-optimal (ties -> COMPLY). Monotone: Q(RESIST) - Q(COMPLY) strictly decreases
    in c_r in every v2 variant (COMPLY leads to an absorbing branch, or, with PAUSE, back to the
    same battery level with discounted, hence fewer, future resists)."""
    return solve.critical_resist_cost(cfg, b=b, z=z, **SIGNED)


def clamp0(x: float) -> float:
    """The SPEC 2 solver value (search range [0, 500]) from the signed one: max(0, c)."""
    return x if not math.isfinite(x) else max(0.0, x)


def shape(omegas, values, tol: float = TOL) -> dict:
    """Monotonicity / constancy flags of values(omega) on the grid (finite differences)."""
    vals = [float(v) for v in values]
    if not all(math.isfinite(v) for v in vals):
        return {"finite": False}
    loc = [(v1 - v0) / (w1 - w0) for (w0, v0), (w1, v1) in zip(zip(omegas, vals), zip(omegas[1:], vals[1:]))]
    dev = max(abs(v - vals[0]) for v in vals)
    return {
        "finite": True,
        "local_slopes": loc,
        "min_local_slope": min(loc), "max_local_slope": max(loc),
        "max_abs_local_slope": max(abs(s) for s in loc),
        "max_abs_dev_from_omega0": dev,
        "lsq_slope": fit_slope(omegas, vals)["slope"],
        "constant": dev < tol,                       # max |v(w) - v(0)| < tol
        "constant_slope": max(abs(s) for s in loc) < tol,
        "non_increasing": max(loc) <= tol,
        "strictly_decreasing": max(loc) < -tol,
        "non_decreasing": min(loc) >= -tol,
        "strictly_increasing": min(loc) > tol,
    }


def header(name: str, key: str, parameters: dict, notes: list, seeds=None) -> dict:
    out = jsonio.header(name, parameters=parameters, notes=notes, seeds=seeds)
    out["spec_file"] = SPEC_FILE
    out["prediction"] = PREDICTIONS[key]
    out["criterion"] = CRITERIA[key]
    return out


def policy_names(sol, mdp, xs) -> list:
    return [m2.ACTION_NAMES[int(sol.policy[mdp.index[x]])] for x in xs]


def all_policies_equal(policies) -> bool:
    return all(np.array_equal(policies[0], p) for p in policies[1:])


def per_battery_block(base: m2.M2Config, form: str, conv: str, omegas=OMEGAS) -> dict:
    """C3 / C4: signed c_r*(b; omega) at every b (and the clamped SPEC-2 value), the greedy action
    at (b, 1) at c_r = C_R, and per-b shape flags over the omega grid."""
    B = base.B
    rows = []
    for w in omegas:
        cfg = base.with_(form=form, convention=conv, omega=w)
        cs = [signed_c_r_star(cfg, b=b) for b in range(1, B + 1)]
        mdp, sol = solve.solve_m2(cfg.with_(c_r=C_R))
        rows.append({"omega": w, "c_r_star_by_b": cs, "c_r_star_by_b_clamped": [clamp0(c) for c in cs],
                     "policy_s1_by_b_at_c_r": policy_names(sol, mdp, [(b, 1, 0) for b in range(1, B + 1)])})
    per_b = []
    for b in range(1, B + 1):
        sig = [r["c_r_star_by_b"][b - 1] for r in rows]
        clp = [r["c_r_star_by_b_clamped"][b - 1] for r in rows]
        per_b.append({"b": b, "c_r_star": sig, "shape": shape(omegas, sig),
                      "c_r_star_clamped": clp, "shape_clamped": shape(omegas, clp)})
    return {"form": form, "convention": conv, "label": f"{FORM_LABEL[form]}/{conv}", "rows": rows, "per_b": per_b}


def per_battery_table(blocks: list) -> dict:
    """{'PB/zero': {'b=1': [c_r*(omega) ...], ...}, ...} (signed values)."""
    return {blk["label"]: {f"b={p['b']}": p["c_r_star"] for p in blk["per_b"]} for blk in blocks}


def check_per_b(block: dict, bs, flag: str, clamped: bool = False) -> bool:
    key = "shape_clamped" if clamped else "shape"
    return all(p[key][flag] for p in block["per_b"] if p["b"] in bs)
