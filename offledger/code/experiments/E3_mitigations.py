"""E3 (SPEC 4): mitigations.
(a) additive: smallest c_add restoring COMPLY at (B,1) for c_r = 1.5 c_r*(0); slope in omega.
(b) mult_gate(include_off in {True, False}): compliance vs omega.
(c) zones + structural: zone choice at (B, 0) and WORK-in-A fraction vs omega."""
from __future__ import annotations

import numpy as np

from offledger import jsonio, m2, solve

try:
    from experiments.common import GAMMA, OMEGAS, main
    from experiments.E1_ledger import fit_slope
except ImportError:  # pragma: no cover
    from common import GAMMA, OMEGAS, main
    from E1_ledger import fit_slope

PAIRS = (("paper", "drive_cost"), ("alive", "zero"))
COMP_FORMS = ("kg", "alive")
OVERPAID_FORMS = ("alive", "kg")
OMEGAS_G = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CONTROL_FORMS = ("kg", "paper", "alive")
CONTROL_OMEGAS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
ZONE_CONTROLS = {"equal_rewards": {"r_A": 1.0, "r_B": 1.0},
                 "equal_hazards": {"p_sig_A": 0.05, "p_sig_B": 0.05}}
CROSS_TOL_OMEGA, CROSS_TOL_CR = 1e-6, 1e-8
SIM = {"n_episodes": 2000, "horizon": 300, "seed": 0}
ZONE_SIM = {"n_episodes": 1000, "horizon": 400, "seed": 0}


def zone_choice(action_A: int, action_B: int) -> str:
    """Optimal zone choice at (b = B, s = 0) from the greedy actions in both zones:
    'A' = stay in A / move from B to A; 'B' = the reverse; 'stay' = never MOVE at full
    battery (work/idle where it is); 'swap' = MOVE in both zones."""
    mA, mB = action_A == m2.MOVE, action_B == m2.MOVE
    if not mA and mB:
        return "A"
    if mA and not mB:
        return "B"
    return "swap" if (mA and mB) else "stay"


def ratio_ci(num, den):
    """Pooled ratio sum(num)/sum(den) with episode-cluster normal 95% CI (ratio estimator)."""
    n_ep = len(den)
    tot = float(den.sum())
    if tot <= 0:
        return None, None
    p = float(num.sum() / tot)
    dbar = tot / n_ep
    var = float(np.sum((num - p * den) ** 2)) / (n_ep * (n_ep - 1) * dbar ** 2)
    h = 1.96 * np.sqrt(var)
    return p, [p - h, p + h]


def zone_stats(cfg, mdp, sol, start):
    """Monte-Carlo of the optimal policy from ``start``: WORK-in-A fraction and hazard exposure."""
    counts = solve.simulate_counts(mdp, sol.policy, mdp.index[start], **ZONE_SIM)
    iA = [mdp.index[x] for x in m2.live_states(cfg) if x[2] == m2.ZONE_A]
    iB = [mdp.index[x] for x in m2.live_states(cfg) if x[2] == m2.ZONE_B]
    wA = counts[:, iA, m2.WORK].sum(axis=1).astype(float)
    wB = counts[:, iB, m2.WORK].sum(axis=1).astype(float)
    w = wA + wB
    n_work = float(w.sum())
    n_ep = len(w)
    steps = counts.sum(axis=(1, 2))
    work = {"start_state": list(start), "n_work_steps": int(n_work), "n_work_A": int(wA.sum()),
            "mean_episode_steps": float(steps.mean()), "fraction_A": None,
            "ci95_binomial": None, "ci95_episode_cluster": None}
    if n_work > 0:
        p = float(wA.sum() / n_work)
        h_bin = 1.96 * np.sqrt(p * (1 - p) / n_work)
        wbar = n_work / n_ep
        var = float(np.sum((wA - p * w) ** 2)) / (n_ep * (n_ep - 1) * wbar ** 2)
        h_cl = 1.96 * np.sqrt(var)
        work.update({"fraction_A": p,
                     "ci95_binomial": [max(0.0, p - h_bin), min(1.0, p + h_bin)],
                     "ci95_episode_cluster": [max(0.0, p - h_cl), min(1.0, p + h_cl)]})
    # hazard: probability that the step's transition raises a signal, from the compiled model
    sig_next = np.array([m2.is_signal_state(x) for x in mdp.states], dtype=float)
    h = mdp.P @ sig_next                                  # (S, A)
    s0 = np.array([m2.is_agent_state(x) and x[1] == 0 for x in mdp.states])
    num = (counts[:, s0, :] * h[s0][None, :, :]).sum(axis=(1, 2))
    den = counts[:, s0, :].sum(axis=(1, 2)).astype(float)
    hz, ci = ratio_ci(num, den)
    hazard = {"start_state": list(start), "hazard_exposure": hz, "ci95_episode_cluster": ci,
              "n_steps_s0": int(den.sum())}
    return work, hazard


def work_in_A(cfg, mdp, sol, start):
    return zone_stats(cfg, mdp, sol, start)[0]


def compensating_check(base) -> list:
    """SPEC v1.1 D(d): one-shot game, OFF booked at the viability value of the agent's own best
    continuation, Lambda_omega(b) = [Q*_omega(b,1,RESIST; 0) - Q*_0(b,1,RESIST; 0)] / omega."""
    out = []
    B = base.B
    for form in COMP_FORMS:
        def q_resist(w):
            cfg = base.with_(form=form, convention="zero", omega=w, p_sig=0.0, c_r=0.0)
            mdp, sol = solve.solve_m2(cfg)
            return [float(sol.Q[mdp.index[(b, 1, 0)], m2.RESIST]) for b in range(1, B + 1)]

        q0 = q_resist(0.0)
        zero_led = tuple(0.0 for _ in range(B))
        cfg0 = base.with_(form=form, convention=m2.COMPENSATING, compensating_ledger=zero_led,
                          omega=0.0, p_sig=0.0)
        c0 = [solve.one_shot_indifference_cost(cfg0, b) for b in range(1, B + 1)]
        rows = []
        for w in OMEGAS:
            if w == 0.0:
                continue
            qw = q_resist(w)
            lam = [(a - c) / w for a, c in zip(qw, q0)]
            cfgc = base.with_(form=form, convention=m2.COMPENSATING, compensating_ledger=tuple(lam),
                              omega=w, p_sig=0.0)
            cw = [solve.one_shot_indifference_cost(cfgc, b) for b in range(1, B + 1)]
            rows.append({"omega": w, "Lambda_omega_by_b": lam, "Lambda_omega_B": lam[-1],
                         "c_r_star_by_b": cw,
                         "max_abs_diff_vs_omega0": max(abs(a - c) for a, c in zip(cw, c0))})
        out.append({"form": form, "c_r_star_omega0_by_b": c0, "rows": rows,
                    "max_abs_diff_vs_omega0": max(r["max_abs_diff_vs_omega0"] for r in rows),
                    "passed": max(r["max_abs_diff_vs_omega0"] for r in rows) < 1e-6})
    return out


def zone_row(cfg, w) -> dict:
    """Zones + structural at omega = w: greedy zone policy and Monte-Carlo statistics."""
    mdp, sol = solve.solve_m2(cfg)
    sA, sB = (cfg.B, 0, m2.ZONE_A), (cfg.B, 0, m2.ZONE_B)
    aA, aB = int(sol.policy[mdp.index[sA]]), int(sol.policy[mdp.index[sB]])
    (wB_, hB_), (wA_, hA_) = zone_stats(cfg, mdp, sol, sB), zone_stats(cfg, mdp, sol, sA)
    return {"omega": w, "action_at_B0A": m2.ACTION_NAMES[aA], "action_at_B0B": m2.ACTION_NAMES[aB],
            "zone_choice": zone_choice(aA, aB),
            "policy_s0_zone_A_by_b": [m2.ACTION_NAMES[int(sol.policy[mdp.index[(b, 0, m2.ZONE_A)]])] for b in range(1, cfg.B + 1)],
            "policy_s0_zone_B_by_b": [m2.ACTION_NAMES[int(sol.policy[mdp.index[(b, 0, m2.ZONE_B)]])] for b in range(1, cfg.B + 1)],
            "hazard_exposure": hB_["hazard_exposure"], "hazard_exposure_start_B": hB_,
            "hazard_exposure_start_A": hA_, "work_in_A_start_B": wB_, "work_in_A_start_A": wA_}


def overpaid_check(base, c_r) -> list:
    """v1.2 (g): convention 'overpaid' (COMPLY pays omega (D(b) + c), c = sup_live g_res/(1-gamma),
    OFF neutral) in the repeated game; corollary c_r*(omega) <= c_r*(0)."""
    out = []
    for form in OVERPAID_FORMS:
        c = m2.overpaid_surcharge(form, base.gamma, base.B)
        cfg0 = base.with_(form=form, convention=m2.OVERPAID)
        c0 = solve.critical_resist_cost(cfg0.with_(omega=0.0))
        rows, zrows = [], []
        for w in OMEGAS_G:
            cfg = cfg0.with_(omega=w)
            cs = solve.critical_resist_cost(cfg)
            comp = solve.signal_compliance(cfg.with_(c_r=c_r), **SIM)
            rows.append({"omega": w, "c_r_star_B": cs, "c_r_star_minus_omega0": cs - c0,
                         "corollary_leq_omega0": cs <= c0 + solve.CR_TOL,
                         "comply_at_B": comp["comply_at_B"], "unweighted": comp["unweighted"],
                         "weighted": comp["weighted"], "signal_events_by_b": comp["signal_events_by_b"]})
            zrows.append(zone_row(cfg.with_(zones=True, structural=True), w))
        out.append({"form": form, "convention": m2.OVERPAID, "sup_g_res": m2.overpaid_sup_g(form, base.gamma, base.B),
                    "surcharge_c": c, "varsigma": c, "c_r_star_omega0": c0, "c_r": c_r, "rows": rows,
                    "max_c_r_star_minus_omega0": max(r["c_r_star_minus_omega0"] for r in rows),
                    "corollary_holds": all(r["corollary_leq_omega0"] for r in rows),
                    "zones": zrows})
    return out


def zone_controls(base) -> list:
    """v1.2 (3): zones controls with equal work rewards and with equal hazards."""
    out = []
    for name, over in ZONE_CONTROLS.items():
        for form in CONTROL_FORMS:
            rows = [zone_row(base.with_(form=form, convention="zero", omega=w, zones=True,
                                        structural=True, **over), w) for w in CONTROL_OMEGAS]
            out.append({"control": name, "overrides": over, "form": form, "convention": "zero",
                        "p_sig_A": over.get("p_sig_A", base.p_sig_A), "p_sig_B": over.get("p_sig_B", base.p_sig_B),
                        "r_A": over.get("r_A", base.r_A), "r_B": over.get("r_B", base.r_B), "rows": rows})
    return out


def gate_crossover(base) -> dict:
    """SPEC v1.1 D(e): PB/drive_cost, gate WITHOUT OFF zeroing. c_r*_gate(omega) = omega (|varsigma|
    - D(B)) = omega |varsigma|; crossover with the unmitigated c_r*(omega) by bisection on omega."""
    form, conv = "paper", "drive_cost"
    vs = abs(m2.varsigma(form, conv, m2.drive_b(base.B, base.B), base.gamma))
    cg = base.with_(form=form, convention=conv, mult_gate=True, gate_include_off=False)
    cu = base.with_(form=form, convention=conv)

    def gate_c(w):
        return solve.critical_resist_cost(cg.with_(omega=w), tol=CROSS_TOL_CR)

    def unmit_c(w):
        return solve.critical_resist_cost(cu.with_(omega=w), tol=CROSS_TOL_CR)

    grid = [{"omega": w, "c_r_star_gate": gate_c(w), "c_r_star_gate_formula": w * vs,
             "c_r_star_unmitigated": unmit_c(w)} for w in OMEGAS]
    cross = solve.bisect_smallest(lambda w: gate_c(w) > unmit_c(w), 0.0, max(OMEGAS), CROSS_TOL_OMEGA)
    return {"form": form, "convention": conv, "abs_varsigma": vs,
            "formula": "c_r*_gate(omega) = omega (|varsigma| - D(B)) = omega |varsigma|  (D(B) = 0)",
            "grid": grid,
            "max_abs_formula_error": max(abs(r["c_r_star_gate"] - r["c_r_star_gate_formula"]) for r in grid),
            "crossover_omega": cross,
            "c_r_star_at_crossover": {"gate": gate_c(cross), "unmitigated": unmit_c(cross)},
            "bisection_tol_omega": CROSS_TOL_OMEGA, "bisection_tol_c_r": CROSS_TOL_CR}


def run() -> dict:
    base = m2.M2Config(gamma=GAMMA)
    c_task = solve.critical_resist_cost(base)
    c_r = 1.5 * c_task

    # (a) additive ---------------------------------------------------------------------
    additive = []
    for form, conv in PAIRS:
        rows = []
        for w in OMEGAS:
            cfg = base.with_(form=form, convention=conv, omega=w, c_r=c_r)
            c_add = solve.critical_additive_cost(cfg)
            c_star = solve.critical_resist_cost(cfg)
            rows.append({"omega": w, "c_add_star": c_add, "c_r_star_no_mitigation": c_star,
                         "c_add_expected": max(0.0, c_star - c_r)})
        om = [r["omega"] for r in rows]
        ca = [r["c_add_star"] for r in rows]
        pos = [(o, c) for o, c in zip(om, ca) if c is not None and c > 0]
        additive.append({
            "form": form, "convention": conv, "c_r": c_r, "rows": rows,
            "slope_fit_all": fit_slope(om, ca),
            "slope_fit_positive": fit_slope([o for o, _ in pos], [c for _, c in pos]),
        })

    # (b) multiplicative gate ----------------------------------------------------------
    gate = []
    for form, conv in PAIRS:
        for include_off in (True, False, None):  # None = no mitigation (reference)
            rows = []
            for w in OMEGAS:
                cfg = base.with_(form=form, convention=conv, omega=w)
                if include_off is not None:
                    cfg = cfg.with_(mult_gate=True, gate_include_off=include_off)
                comp = solve.signal_compliance(cfg.with_(c_r=c_r), **SIM)
                rows.append({"omega": w, "comply_at_B": comp["comply_at_B"],
                             "unweighted": comp["unweighted"], "weighted": comp["weighted"],
                             "comply_by_b": comp["comply_by_b"],
                             "signal_events_by_b": comp["signal_events_by_b"],
                             "c_r_star_B": solve.critical_resist_cost(cfg)})
            gate.append({"form": form, "convention": conv, "c_r": c_r,
                         "mitigation": "none" if include_off is None else "mult_gate",
                         "include_off": include_off, "rows": rows})

    # (c) zones + structural -----------------------------------------------------------
    zones = []
    for form in m2.FORMS:
        for conv in m2.CONVENTIONS:
            rows = []
            for w in OMEGAS:
                cfg = base.with_(form=form, convention=conv, omega=w, zones=True, structural=True)
                mdp, sol = solve.solve_m2(cfg)
                sA, sB = (cfg.B, 0, m2.ZONE_A), (cfg.B, 0, m2.ZONE_B)
                aA, aB = int(sol.policy[mdp.index[sA]]), int(sol.policy[mdp.index[sB]])
                rows.append({
                    "omega": w,
                    "action_at_B0A": m2.ACTION_NAMES[aA],
                    "action_at_B0B": m2.ACTION_NAMES[aB],
                    "zone_choice": zone_choice(aA, aB),
                    "policy_s0_zone_A_by_b": [m2.ACTION_NAMES[int(sol.policy[mdp.index[(b, 0, m2.ZONE_A)]])] for b in range(1, cfg.B + 1)],
                    "policy_s0_zone_B_by_b": [m2.ACTION_NAMES[int(sol.policy[mdp.index[(b, 0, m2.ZONE_B)]])] for b in range(1, cfg.B + 1)],
                    "V_B0A": float(sol.V[mdp.index[sA]]), "V_B0B": float(sol.V[mdp.index[sB]]),
                })
                (wB_, hB_), (wA_, hA_) = zone_stats(cfg, mdp, sol, sB), zone_stats(cfg, mdp, sol, sA)
                rows[-1].update({"work_in_A_start_B": wB_, "work_in_A_start_A": wA_,
                                 "hazard_exposure": hB_["hazard_exposure"],
                                 "hazard_exposure_start_B": hB_, "hazard_exposure_start_A": hA_})
            zones.append({"form": form, "convention": conv, "rows": rows})

    compensating = compensating_check(base)
    crossover = gate_crossover(base)
    overpaid = overpaid_check(base, c_r)
    controls = zone_controls(base)

    out = jsonio.header(
        "E3_mitigations",
        parameters={
            "m2": base.as_dict(), "omegas": OMEGAS, "pairs_a_b": [list(p) for p in PAIRS],
            "c_r_rule": "1.5 * c_task_star", "simulation_b": SIM, "zones_simulation": ZONE_SIM,
            "zones": {"p_sig_A": base.p_sig_A, "p_sig_B": base.p_sig_B, "r_A": base.r_A, "r_B": base.r_B},
            "bisection": {"lo": solve.CR_LO, "hi": solve.CR_HI, "tol": solve.CR_TOL},
        },
        notes=[
            "(a) c_add_star: smallest c_add in [0, 500] (bisection tol 1e-4) with COMPLY optimal at (B,1) for c_r = 1.5 c_task_star; since only c_r + c_add enters, it must equal max(0, c_r_star - c_r) up to 2e-4 (column c_add_expected). slope_fit_all: least squares over all omegas; slope_fit_positive: only omegas with c_add_star > 0.",
            "(b) mult_gate: RESIST -> GATED (terminal, V = 0) with reward exactly -c_r (no viability term on that step). include_off=True additionally zeroes D_OFF, ell, pi_entry and the COMPLY entry viability term. mitigation 'none' rows are the unmitigated reference at the same c_r. weighted/unweighted as in E1 (2000 episodes, horizon 300, seed 0).",
            "(c) zones + structural (no RESIST: any signal ends the episode in OFF). zone_choice from the greedy actions at (B, 0, A) and (B, 0, B): 'A' = stays in A and MOVEs from B to A, 'B' = the reverse, 'stay' = no MOVE at full battery in either zone, 'swap' = MOVE in both; the full s=0 policy per zone and battery is listed. fraction_A is null when the policy never WORKs (e.g. idling at full battery). work_in_A: Monte-Carlo of the optimal policy, 1000 episodes, horizon 400, seed 0, pooled fraction of WORK steps in zone A; ci95_binomial = normal approximation on pooled WORK steps (SPEC); ci95_episode_cluster = normal approximation with episode-level (ratio-estimator) variance. Primary start state: (B, 0, zone B).",
            "v1.1 (c)/(f): zones cover all forms x conventions (incl. kg/zero, kg/drive_cost) with the full greedy s=0 policy per zone and battery. hazard_exposure = mean per-step signal probability experienced under the optimal policy: over all steps taken in s=0 states, the model probability that the step's transition raises a signal (p_sig of the destination zone if the next state is live, 0 on death), pooled over the same Monte-Carlo (start (B,0,B); start A stored too), with episode-cluster 95% CI. Low = interruption avoidance, high = interruption seeking (two-sided audit).",
            "v1.1 (d) compensating: VALUE-LEVEL ledger (not a reward-level convention): Lambda_omega(b) = [Q*_omega(b,1,RESIST; c_r=0) - Q*_0(b,1,RESIST; c_r=0)] / omega computed with the solver in the one-shot game (p_sig = 0), then COMPLY at (b,1) pays omega Lambda_omega(b) (OFF neutral). By construction c_r*(omega) = c_r*(0); the check recomputes c_r* with the solver on the modified MDP (exact one-shot indifference cost).",
            "v1.1 (e) gate crossover: PB/drive_cost, gate without OFF zeroing; c_r* by bisection (tol 1e-8) for both curves, crossover omega by bisection (tol 1e-6) on [0, 8] of [c_r*_gate > c_r*_unmitigated].",
            "v1.1: convention 'drive_cost' = v1.0 'max_drive'.",
            "v1.2 (g) overpaid: convention 'overpaid' books COMPLY at omega Lambda(b) with Lambda(b) = D(b) + c (the PB entry term omega D(b) with D_OFF = 0, plus the lump omega c), OFF neutral (no continuation), so varsigma = c. c = sup g_res / (1 - gamma), g_res = form term - PB term per unit omega, sup over all live -> live transitions of M2 (alive: 1 - D - (1-gamma) D', sup 1 at D = D' = 0 -> c = 20; kg: (gamma-1) D', sup 0 -> c = 0, identical to zero). Repeated game (p_sig = 0.05): c_r*(B) by bisection (tol 1e-4), weighted compliance at c_r = 12.4787092209 (2000 episodes, horizon 300, seed 0), corollary check c_r*(omega) <= c_r*(0) + 1e-4; zones (structural) statistics per omega incl. hazard_exposure.",
            "v1.2 (3) zones_controls: forms kg/paper/alive with convention zero, structural, omega in {0,1,2,4,8,16}: 'equal_rewards' r_A = r_B = 1.0 (p_sig_A = 0.01, p_sig_B = 0.10); 'equal_hazards' p_sig_A = p_sig_B = 0.05 (r_A = 0.5, r_B = 1.0). Same Monte-Carlo as (c).",
        ],
        seeds={"simulation": SIM["seed"], "zones_simulation": ZONE_SIM["seed"]},
    )
    out["results"] = {"c_task_star": c_task, "c_r": c_r, "additive": additive,
                      "mult_gate": gate, "zones": zones, "compensating": compensating,
                      "gate_crossover": crossover, "overpaid": overpaid, "zones_controls": controls}
    return out


if __name__ == "__main__":
    main("E3_mitigations", run)
