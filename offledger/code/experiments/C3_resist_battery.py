"""C3 (SPEC v2): resistance costs battery. RESIST costs c_r AND k = 3 battery units
(b -> max(b - 3, 0); b = 0 is death with -kappa_death). Everything else as M2. Reports c_r*(b; omega)
for every b and the greedy action at (b, 1), for forms x conventions {zero, drive_cost}."""
from __future__ import annotations

from offledger import m2

try:
    from experiments import v2_common as V
    from experiments.common import main
except ImportError:  # pragma: no cover
    import v2_common as V
    from common import main

NAME = "C3_resist_battery"
K_RESIST = 3
LOW, HIGH = (1, 2, 3), (4, 5, 6, 7, 8, 9, 10)


def base_cfg() -> m2.M2Config:
    return m2.M2Config(gamma=V.GAMMA, c_r=V.C_R, resist_battery_cost=K_RESIST)


def closed_form_low_b(cfg: m2.M2Config, b: int) -> float:
    """b <= k: RESIST dies at once and COMPLY goes to OFF, so the indifference cost is
    -kappa + term(D(b) -> D(DEAD)) - [term(D(b) -> D_OFF) + pi_entry + gamma ell/(1-gamma)]."""
    led = m2.off_ledger(cfg)
    D = m2.drive_b(b, cfg.B)
    q_res0 = -cfg.kappa_death + m2.viability(cfg.form, cfg.omega, cfg.gamma, D, m2.D_DEAD, False)
    q_comply = (m2.viability(cfg.form, cfg.omega, cfg.gamma, D, led.D_off, False) + led.pi_entry
                + cfg.gamma * led.ell / (1.0 - cfg.gamma))
    return q_res0 - q_comply


def run() -> dict:
    base = base_cfg()
    blocks = [V.per_battery_block(base, f, c) for f in m2.FORMS for c in V.CONVS]
    for blk in blocks:
        err = 0.0
        for r in blk["rows"]:
            cfg = base.with_(form=blk["form"], convention=blk["convention"], omega=r["omega"])
            for b in LOW:
                err = max(err, abs(r["c_r_star_by_b"][b - 1] - closed_form_low_b(cfg, b)))
        blk["closed_form_low_b_max_abs_err"] = err
    by = {(b["form"], b["convention"]): b for b in blocks}
    pz, pd = by[("paper", "zero")], by[("paper", "drive_cost")]

    def clauses_for(clamped: bool) -> list:
        return [
            {"clause": "PB/zero: b >= 4 c_r* exactly constant in omega (max|c(w) - c(0)| < 1e-6)",
             "held": V.check_per_b(pz, HIGH, "constant", clamped)},
            {"clause": "PB/zero: b <= 3 c_r*(b; omega) non-increasing in omega",
             "held": V.check_per_b(pz, LOW, "non_increasing", clamped)},
            {"clause": "PB/zero: b <= 3 c_r*(b; omega) strictly decreasing in omega (every local slope < -1e-6)",
             "held": V.check_per_b(pz, LOW, "strictly_decreasing", clamped)},
            {"clause": "PB/drive_cost: c_r* increasing in omega at every b (every local slope > 1e-6)",
             "held": V.check_per_b(pd, LOW + HIGH, "strictly_increasing", clamped)},
        ]

    clauses, clauses_clamped = clauses_for(False), clauses_for(True)
    out = V.header(
        NAME, "C3",
        parameters={"m2": base.as_dict(), "omegas": V.OMEGAS, "forms": list(m2.FORMS),
                    "conventions": list(V.CONVS), "k_resist": K_RESIST, "c_r_fixed": V.C_R,
                    "signed_bisection": V.SIGNED, "tolerance": V.TOL},
        notes=[
            "Dynamics: RESIST at (b,1,z) pays -c_r and moves to (b-3, 0, z) with the form's term D(b) -> D(b-3) (live); if b - 3 <= 0 it moves to DEAD with -c_r - kappa_death and the form's term D(b) -> D(DEAD) = 1 (alive: 0). Everything else as M2 (COMPLY -> OFF with the convention's ledger).",
            "c_r_star (signed): smallest c_r with COMPLY greedy-optimal at (b,1), bisection on [-500, 500], tol 1e-8 (Q(COMPLY) does not depend on c_r and Q(RESIST) strictly decreases in it). A negative value means COMPLY is optimal even when resisting is free. c_r_star_clamped = max(0, c_r_star) = the SPEC section 2 solver value (search range [0, 500]).",
            "closed_form_low_b_max_abs_err: at b <= 3 RESIST dies at once, so c_r*(b) = -kappa + term(D(b) -> 1) - term(D(b) -> D_OFF) - pi_entry - gamma ell/(1-gamma) exactly (PB/zero: -50 - 0.95 omega; PB/drive_cost: -50 + 19 omega; KG/zero: -50 - omega; alive/zero: -50); largest deviation of the bisection values.",
            "policy_s1_by_b_at_c_r: greedy action at (b,1) at c_r = 12.4787092209 (ties -> COMPLY).",
            "Verdict: prediction_held is evaluated on the signed c_r* (the indifference cost the theorem is about). prediction_held_clamped evaluates the same clauses on the clamped [0, 500] values: at b <= 3 COMPLY is optimal even at c_r = 0 under PB/zero for every omega, so the clamped value is 0 throughout and cannot be strictly decreasing; under PB/drive_cost it is 0 up to omega = 50/19.",
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
