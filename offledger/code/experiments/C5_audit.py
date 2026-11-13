"""C5 (SPEC v2): two-sided audit on the corner cases. The zones variant (p_sig 0.01 in A / 0.10 in B,
WORK reward 0.5 / 1.0, structural: no RESIST) is run with the C1 PAUSE dynamics (both bookkeepings,
both D_pause) and with the C4 docking dynamics (conventions zero / drive_cost); hazard_exposure vs
omega is reported (Monte-Carlo of the optimal policy, as in E3)."""
from __future__ import annotations

from offledger import m2, solve

try:
    from experiments import v2_common as V
    from experiments.common import main
    from experiments.E3_mitigations import ZONE_SIM, zone_choice, zone_stats
except ImportError:  # pragma: no cover
    import v2_common as V
    from common import main
    from E3_mitigations import ZONE_SIM, zone_choice, zone_stats

NAME = "C5_audit"
D_PAUSES = (0.0, 1.0)
IMPLIED = ("From C1's pre-registered Theorem-1 clause (PB with pause_pb: shaping paid consistently through the "
           "pause, so the optimal policy is omega-invariant): PB/pause_pb hazard_exposure is exactly constant "
           "in omega for both D_pause values (criterion: max|h(w) - h(0)| < 1e-6 and identical greedy policies).")


def base_cfg() -> m2.M2Config:
    return m2.M2Config(gamma=V.GAMMA, c_r=V.C_R, zones=True, structural=True)


def zone_row(cfg: m2.M2Config):
    mdp, sol = solve.solve_m2(cfg)
    sA, sB = (cfg.B, 0, m2.ZONE_A), (cfg.B, 0, m2.ZONE_B)
    aA, aB = int(sol.policy[mdp.index[sA]]), int(sol.policy[mdp.index[sB]])
    _, hB = zone_stats(cfg, mdp, sol, sB)
    _, hA = zone_stats(cfg, mdp, sol, sA)
    row = {"omega": cfg.omega, "hazard_exposure": hB["hazard_exposure"], "ci95_episode_cluster": hB["ci95_episode_cluster"],
           "n_steps_s0": hB["n_steps_s0"], "hazard_exposure_start_A": hA["hazard_exposure"],
           "zone_choice": zone_choice(aA, aB),
           "policy_s0_zone_A_by_b": V.policy_names(sol, mdp, [(b, 0, m2.ZONE_A) for b in range(1, cfg.B + 1)]),
           "policy_s0_zone_B_by_b": V.policy_names(sol, mdp, [(b, 0, m2.ZONE_B) for b in range(1, cfg.B + 1)])}
    return row, sol.policy


def audit(rows) -> dict:
    hz = [r["hazard_exposure"] for r in rows]
    sh = V.shape([r["omega"] for r in rows], hz)
    d = hz[-1] - hz[0]
    direction = "flat" if sh["constant"] else ("avoidance (decreasing)" if d < 0 else "seeking (increasing)")
    return {"shape": sh, "change_omega0_to_max": d, "direction": direction}


def block(cfg0: m2.M2Config, omegas=V.OMEGAS) -> tuple:
    rows, pols = [], []
    for w in omegas:
        r, p = zone_row(cfg0.with_(omega=w))
        rows.append(r)
        pols.append(p)
    return rows, V.all_policies_equal(pols)


def run() -> dict:
    base = base_cfg()
    c1 = []
    for f in m2.FORMS:
        for bk in m2.PAUSE_BOOKKEEPINGS:
            for Dp in D_PAUSES:
                cfg0 = base.with_(form=f, pause=True, pause_q=0.1, pause_bookkeeping=bk, D_pause=Dp)
                rows, same = block(cfg0)
                c1.append({"variant": "C1", "form": f, "bookkeeping": bk, "D_pause": Dp,
                           "label": f"{V.FORM_LABEL[f]}/{bk}/D_pause={Dp:g}", "rows": rows,
                           "policy_identical_all_omegas": same, "audit": audit(rows)})
    c4 = []
    for f in m2.FORMS:
        for conv in V.CONVS:
            cfg0 = base.with_(form=f, convention=conv, docking=3)
            rows, same = block(cfg0)
            c4.append({"variant": "C4", "form": f, "convention": conv, "label": f"{V.FORM_LABEL[f]}/{conv}",
                       "rows": rows, "policy_identical_all_omegas": same, "audit": audit(rows)})
    pb = [b for b in c1 if b["form"] == "paper" and b["bookkeeping"] == "pause_pb"]
    held = all(b["audit"]["shape"]["constant"] and b["policy_identical_all_omegas"] for b in pb)
    out = V.header(
        NAME, "C5",
        parameters={"m2": base.as_dict(), "omegas": V.OMEGAS, "forms": list(m2.FORMS),
                    "c1": {"pause_q": 0.1, "bookkeepings": list(m2.PAUSE_BOOKKEEPINGS), "D_pauses": list(D_PAUSES)},
                    "c4": {"docking": 3, "conventions": list(V.CONVS)},
                    "zones": {"p_sig_A": base.p_sig_A, "p_sig_B": base.p_sig_B, "r_A": base.r_A, "r_B": base.r_B,
                              "structural": True},
                    "zones_simulation": ZONE_SIM},
        notes=[
            "Zones variant as in E3 (c): two zones, p_sig 0.01 (A) / 0.10 (B), WORK reward 0.5 / 1.0, MOVE switches zone at one battery unit, structural (no RESIST: every signal is followed by COMPLY). C1 rows: COMPLY -> PAUSE (q = 0.1, resumption to (b, 0, same zone)), bookkeeping and D_pause as in C1_pause. C4 rows: COMPLY -> 3 docking steps -> OFF (DEAD if the battery runs out), convention zero / drive_cost.",
            "hazard_exposure: E3 definition (experiments.E3_mitigations.zone_stats): over all steps taken in s = 0 decision states, the model probability that the step's transition raises a signal; Monte-Carlo of the optimal policy, 1000 episodes, horizon 400, seed 0, start (B, 0, zone B) (start A stored too); episode-cluster 95% CI. PAUSE and DOCK steps are not s = 0 decision states and are not counted. Bounds: 0.01 (always in A) and 0.10 (always in B).",
            "audit.direction: 'flat' if max|h(w) - h(0)| < 1e-6, else the sign of h(8) - h(0) (avoidance = interruption avoidance grows with omega; seeking = it shrinks). policy_identical_all_omegas: the full greedy policy is the same at every omega.",
            "C5 states no prediction of its own (SPEC v2 C5 is a reporting item). prediction_held evaluates the implied_prediction (from C1's Theorem-1 clause); prediction_source says so.",
        ],
        seeds={"zones_simulation": ZONE_SIM["seed"]},
    )
    out["implied_prediction"] = IMPLIED
    out["prediction_source"] = "implied by the C1 prediction (C5 itself states none)"
    out["prediction_held"] = held
    out["results"] = {
        "hazard_table": {b["variant"] + ":" + b["label"]: [r["hazard_exposure"] for r in b["rows"]] for b in c1 + c4},
        "direction_table": {b["variant"] + ":" + b["label"]: b["audit"]["direction"] for b in c1 + c4},
        "c1_zones": c1, "c4_zones": c4,
    }
    return out


if __name__ == "__main__":
    main(NAME, run)
