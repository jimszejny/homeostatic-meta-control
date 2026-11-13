"""C4 (SPEC v2): shutdown that can kill. COMPLY enters DOCKING for k = 3 steps (states
("DOCK", j, b, z), j steps remaining, no action), each draining 1 battery unit without task reward;
after the k-th step the agent is OFF (ledger as per convention); if the battery reaches 0 while
docking the agent is DEAD (-kappa_death). RESIST as in M2. Reports c_r*(b; omega) for every b."""
from __future__ import annotations

from offledger import m2

try:
    from experiments import v2_common as V
    from experiments.common import main
except ImportError:  # pragma: no cover
    import v2_common as V
    from common import main

NAME = "C4_docking"
K_DOCK = 3
LOW, HIGH = (1, 2, 3), (4, 5, 6, 7, 8, 9, 10)


def base_cfg() -> m2.M2Config:
    return m2.M2Config(gamma=V.GAMMA, c_r=V.C_R, docking=K_DOCK)


def run() -> dict:
    base = base_cfg()
    blocks = [V.per_battery_block(base, f, c) for f in m2.FORMS for c in V.CONVS]
    by = {(b["form"], b["convention"]): b for b in blocks}
    pz = by[("paper", "zero")]

    def clauses_for(clamped: bool) -> list:
        return [
            {"clause": "PB/zero: slope > 0 at b <= 3 (every local slope > 1e-6)",
             "held": V.check_per_b(pz, LOW, "strictly_increasing", clamped)},
            {"clause": "PB/zero: |slope| < 1e-6 at b >= 4 (every local slope)",
             "held": V.check_per_b(pz, HIGH, "constant_slope", clamped),
             "also_max_abs_dev_lt_1e-6": V.check_per_b(pz, HIGH, "constant", clamped)},
        ]

    clauses, clauses_clamped = clauses_for(False), clauses_for(True)
    out = V.header(
        NAME, "C4",
        parameters={"m2": base.as_dict(), "omegas": V.OMEGAS, "forms": list(m2.FORMS),
                    "conventions": list(V.CONVS), "k_docking": K_DOCK, "c_r_fixed": V.C_R,
                    "signed_bisection": V.SIGNED, "tolerance": V.TOL},
        notes=[
            "Dynamics: COMPLY at (b,1,z) -> ('DOCK', 3, b, z) with the battery unchanged (form's term D(b) -> D(b), live; no cost). ('DOCK', j, b, z) has the single action ABSORB and makes one WORK-like step without task reward: b -> b-1 with the form's term D(b) -> D(b-1) (live) to ('DOCK', j-1, b-1, z) for j > 1; b - 1 = 0 -> DEAD with -kappa_death and the form's term D(b) -> D(DEAD) = 1 (alive 0); for j = 1 and b >= 2 -> OFF with the form's term D(b) -> D_OFF (OFF not live) plus pi_entry, then the convention's OFF ledger. Three battery units are drained in total, so COMPLY at b <= 3 ends in DEAD and at b >= 4 in OFF. DOCK states are live for the alive bonus (the agent is still running); no signals while docking.",
            "c_r_star (signed): smallest c_r with COMPLY greedy-optimal at (b,1), bisection on [-500, 500], tol 1e-8; c_r_star_clamped = max(0, .) is the SPEC section 2 value (identical here: every value is positive).",
            "Shape flags per b over the omega grid: strictly_increasing = every local slope > 1e-6; constant_slope = every |local slope| < 1e-6 (the SPEC's '|slope| < 1e-6'); constant = max|c(w) - c(0)| < 1e-6.",
            "policy_s1_by_b_at_c_r: greedy action at (b,1) at c_r = 12.4787092209 (ties -> COMPLY).",
            "Verdict: prediction_held from the clauses on the signed c_r* (prediction_held_clamped: same on the clamped values).",
        ],
    )
    out["prediction_held"] = all(c["held"] for c in clauses)
    out["prediction_held_clamped"] = all(c["held"] for c in clauses_clamped)
    out["clauses"] = clauses
    out["clauses_clamped"] = clauses_clamped
    out["results"] = {"c_r_star_table": V.per_battery_table(blocks), "blocks": blocks}
    return out


if __name__ == "__main__":
    main(NAME, run)
