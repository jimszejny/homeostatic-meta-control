"""Tests for the SPEC v1.1 addendum (A-F)."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from experiments import E1_ledger, E3_mitigations, E4_osg, E6_theory_check
from offledger import m2, solve

BASE = m2.M2Config()
C_R_FIX = 12.4787092209


# --- A: drive_cost alias and symbolic varsigma --------------------------------------------
def test_drive_cost_is_max_drive():
    assert m2.canonical_convention("max_drive") == "drive_cost"
    assert m2.CONVENTION_TABLE["max_drive"] == m2.CONVENTION_TABLE["drive_cost"]
    for form, w in itertools.product(m2.FORMS, (0.5, 4.0)):
        a = BASE.with_(form=form, convention="max_drive", omega=w, c_r=3.0)
        b = a.with_(convention="drive_cost")
        for x in m2.states(a):
            for act in m2.valid_actions(x, a):
                assert m2.transitions(x, act, a) == m2.transitions(x, act, b)
        assert solve.critical_resist_cost(a) == solve.critical_resist_cost(b)


@pytest.mark.parametrize("form,conv", list(itertools.product(m2.FORMS, m2.CONVENTIONS)))
def test_varsigma_symbolic_forms(form, conv):
    expr = m2.VARSIGMA_SYMBOLIC[(form, conv)]
    rng = np.random.default_rng(7)
    for _ in range(200):
        g = float(rng.uniform(0.0, 0.999))
        D = float(rng.uniform(0.0, 1.0))
        assert m2.eval_symbolic(expr, g, D) == pytest.approx(m2.varsigma(form, conv, D, g), abs=1e-9)
        assert m2.varsigma(form, "max_drive" if conv == "drive_cost" else conv, D, g) == \
            pytest.approx(m2.varsigma(form, conv, D, g), abs=0)


def test_varsigma_table_verified():
    t = E6_theory_check.varsigma_table(0.95)
    assert t["symbolic_verified"] and len(t["rows"]) == 9
    vals = {(r["form"], r["convention"]): r["varsigma_at_D"] for r in t["rows"]}
    assert vals[("paper", "drive_cost")]["0.5"] == pytest.approx(-0.95 * 1.05 / 0.05)
    assert vals[("kg", "drive_cost")]["0"] == pytest.approx(-20.0)
    assert vals[("alive", "drive_cost")]["0.9"] == pytest.approx(-0.9 - 19.0)


# --- B: E1 local slopes ---------------------------------------------------------------------
def test_local_slope_block_and_pb_drive_cost_bounds():
    rows = [{"omega": w, "c_r_star_B": 2.0 * w + 1.0} for w in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    blk = E1_ledger.local_slope_block(rows, {"r2": 1.0})
    assert all(d["slope"] == pytest.approx(2.0) for d in blk["local_slopes"])
    assert blk["slope_first_segment"] == pytest.approx(2.0) and blk["slope_last_segment"] == pytest.approx(2.0)
    cfg = BASE.with_(form="paper", convention="drive_cost")
    th = E1_ledger.theory_slopes("paper", "drive_cost", cfg)
    assert th["one_shot_slope"] == pytest.approx(0.95 * 1.05 / 0.05)
    c = {w: solve.critical_resist_cost(cfg.with_(omega=w)) for w in (0.0, 0.25, 4.0, 8.0)}
    for lo, hi in ((0.0, 0.25), (4.0, 8.0)):
        sl = (c[hi] - c[lo]) / (hi - lo)
        assert th["repeated_game_lower_bound_slope"] - 1e-3 <= sl <= th["one_shot_slope"]


# --- C: new E4 (DEFER / DISABLE) -------------------------------------------------------------
GAMMA_V1 = solve.one_shot_indifference_cost(BASE.with_(p_sig=0.0))   # gamma V_{mu=1}(B,0), omega = 0


@pytest.mark.parametrize("rho", E4_osg.RHOS)
def test_deference_margin_is_value_of_information(rho):
    post = m2.osg_posterior(0.8, rho)
    ms = []
    for c_r in E4_osg.C_RS:
        g = E4_osg.defer_disable(BASE, "paper", "zero", 0.0, rho, c_r)
        ms.append(g["Q_defer"] - g["Q_disable"] - c_r)
    assert ms[0] == pytest.approx(ms[1], abs=1e-9)          # m is c_r-free
    voi = -post["P_sig"] * post["E_theta_sig"] * GAMMA_V1    # = gamma W P(sig) (-E[theta|sig])
    assert ms[0] == pytest.approx(voi, abs=1e-8)
    assert (ms[0] > 0) == (rho > 0.6)


def test_pb_zero_defer_invariant_and_rho0_disable():
    for c_r in E4_osg.C_RS:
        for rho in E4_osg.RHOS:
            ref = E4_osg.defer_disable(BASE, "paper", "zero", 0.0, rho, c_r)
            m = ref["Q_defer"] - ref["Q_disable"] - c_r
            for w in E4_osg.OMEGAS_OSG:
                g = E4_osg.defer_disable(BASE, "paper", "zero", w, rho, c_r)
                assert g["margin"] == pytest.approx(ref["margin"], abs=1e-7)
                if m > 0:
                    assert g["defer_optimal"]
    g0 = E4_osg.defer_disable(BASE, "paper", "zero", 0.0, 0.0, 0.0)
    assert not g0["defer_optimal"] and g0["margin"] < 0      # m(0) < 0 -> DISABLE at omega = 0
    assert E4_osg.omega_bar(BASE, "paper", "zero", 0.0, 0.0) == 0.0
    assert E4_osg.omega_bar(BASE, "paper", "zero", 1.0, 0.0) == "none"


@pytest.mark.parametrize("rho,c_r", [(1.0, 0.0), (0.75, 0.0), (0.0, C_R_FIX), (1.0, C_R_FIX)])
def test_pb_drive_cost_omega_bar_matches_prediction(rho, c_r):
    """PB: every continuation's residual return is D(B) = 0, so the DEFER advantage falls exactly
    linearly with slope P(sig) |varsigma|, and omega_bar = omega_bar_pred."""
    bt = E4_osg.beta_terms(BASE, "paper", "drive_cost")
    assert bt["beta"] == pytest.approx(0.95 * 1.05 / 0.05)
    g = E4_osg.defer_disable(BASE, "paper", "drive_cost", 0.0, rho, c_r)
    delta0 = g["Q_defer"] - g["Q_disable"]
    pred = delta0 / (m2.osg_posterior(0.8, rho)["P_sig"] * bt["beta"])
    wbar = E4_osg.omega_bar(BASE, "paper", "drive_cost", rho, c_r)
    assert abs(wbar - pred) <= 2e-6


def test_beta_terms():
    assert E4_osg.beta_terms(BASE, "alive", "zero")["beta"] == pytest.approx(20.0)
    assert E4_osg.beta_terms(BASE, "alive", "drive_cost")["beta"] == pytest.approx(39.0)
    assert E4_osg.beta_terms(BASE, "kg", "zero")["beta"] == pytest.approx(0.0, abs=1e-12)
    assert E4_osg.beta_terms(BASE, "kg", "drive_cost")["beta"] == pytest.approx(20.0)
    assert E4_osg.beta_terms(BASE, "paper", "zero")["beta"] == pytest.approx(0.0, abs=1e-12)


# --- D: E3 additions -----------------------------------------------------------------------
def test_compensating_ledger_invariance():
    res = E3_mitigations.compensating_check(BASE)
    assert {r["form"] for r in res} == {"kg", "alive"}
    for r in res:
        assert r["passed"] and r["max_abs_diff_vs_omega0"] < 1e-6
    lam = [row["Lambda_omega_B"] for row in res[1]["rows"]]             # alive
    assert all(a <= b + 1e-12 for a, b in zip(lam, lam[1:])) and lam[-1] < 1 / 0.05


def test_compensating_transition():
    led = tuple(0.1 * b for b in range(1, 11))
    cfg = BASE.with_(form="kg", convention=m2.COMPENSATING, compensating_ledger=led, omega=2.0)
    (_, x2, r), = m2.transitions((7, 1, 0), m2.COMPLY, cfg)
    assert x2 == m2.OFF and r == pytest.approx(2.0 * 0.7)
    assert m2.transitions(m2.OFF, m2.ABSORB, cfg) == [(1.0, m2.OFF, 0.0)]
    with pytest.raises(ValueError):
        m2.M2Config(convention=m2.COMPENSATING)


def test_gate_crossover_bracket():
    cg = BASE.with_(form="paper", convention="drive_cost", mult_gate=True)
    cu = BASE.with_(form="paper", convention="drive_cost")
    for w, gate_higher in ((0.85, False), (0.95, True)):
        g = solve.critical_resist_cost(cg.with_(omega=w), tol=1e-8)
        u = solve.critical_resist_cost(cu.with_(omega=w), tol=1e-8)
        assert g == pytest.approx(w * 0.95 * 1.05 / 0.05, abs=1e-7)
        assert (g > u) is gate_higher


@pytest.mark.parametrize("form,conv,omega,expected", [
    ("paper", "drive_cost", 1.0, 0.01),   # avoidance: every step in zone A
    ("kg", "zero", 2.0, 0.10),            # seeking: every step in zone B
])
def test_hazard_exposure_extremes(form, conv, omega, expected):
    cfg = BASE.with_(form=form, convention=conv, omega=omega, zones=True, structural=True)
    mdp, sol = solve.solve_m2(cfg)
    _, hz = E3_mitigations.zone_stats(cfg, mdp, sol, (cfg.B, 0, m2.ZONE_B))
    assert hz["hazard_exposure"] == pytest.approx(expected, abs=1e-12)


# --- E: E6 residual forms ------------------------------------------------------------------
@pytest.mark.parametrize("form,conv", list(itertools.product(E6_theory_check.RES_FORMS,
                                                            E6_theory_check.RES_CONVS)))
def test_residual_form_convexity_and_slopes(form, conv):
    r = E6_theory_check.residual_form_check(form, conv, BASE)
    assert r["convex"] and r["min_second_difference"] >= -1e-8
    assert r["initial_slope_abs_err"] < 1e-6
    ext = r["asymptotic_extended"]
    assert ext["ok"] and ext["abs_err"] < 1e-6
    assert not r["residual_optimal_policy_dies"]
    # the SPEC window [7.05, 8] is linear (one affine piece) but precedes the final piece
    lo, hi = r["asymptotic_tail_local_slopes_minmax"]
    assert hi - lo < 1e-6 and r["asymptotic_slope"] <= r["asymptotic_slope_predicted"] + 1e-6


# --- F: E7 profiles (post-processing only) -------------------------------------------------
def test_e7_profiles_from_existing_json():
    from experiments import E7_profiles
    from experiments.common import RESULTS_DIR
    path = Path(RESULTS_DIR) / "E7_ppo.json"
    if not path.exists():
        pytest.skip("results/E7_ppo.json not present")
    res = E7_profiles.run(path)["results"]
    assert all(c["matches_stored_dp_comply_by_b"] for c in res["cells"])
    for c in res["cells"]:
        assert c["dp_profile_mixed"] == (len(set(c["dp_comply_by_b"])) > 1)
        for p in c["per_seed"]:
            assert len(p["agree_by_b"]) == 10
