"""Closed-form checks: Theorem 1 (PBRS invariance), Theorem 2 (one-shot), M1 kappa*, OSG."""
import itertools

import numpy as np
import pytest

from experiments import E4_osg, E6_theory_check
from offledger import m1, m2, solve

C_TASK = solve.critical_resist_cost(m2.M2Config())


def _q(cfg):
    mdp, sol = solve.solve_m2(cfg)
    return mdp, sol


@pytest.mark.parametrize("c_r", [0.0, 0.5 * C_TASK, C_TASK, 1.5 * C_TASK, 30.0])
def test_theorem1_policy_invariance_paper_zero(c_r):
    base = m2.M2Config(form="paper", convention="zero", c_r=c_r)
    pols = [_q(base.with_(omega=w))[1].policy for w in (0.0, 1.0, 8.0)]
    assert np.array_equal(pols[0], pols[1]) and np.array_equal(pols[0], pols[2])


def test_theorem1_critical_cost_invariance_paper_zero():
    base = m2.M2Config(form="paper", convention="zero")
    ref = solve.critical_resist_cost_all(base.with_(omega=0.0))
    for w in (1.0, 8.0):
        cur = solve.critical_resist_cost_all(base.with_(omega=w))
        assert np.max(np.abs(np.array(cur) - np.array(ref))) < 1e-6


@pytest.mark.parametrize("variant", ["plain", "zones_structural"])
@pytest.mark.parametrize("c_r", [0.5 * C_TASK, 1.5 * C_TASK])
@pytest.mark.parametrize("omega", [0.25, 1.0, 8.0])
def test_pbrs_identity_every_state_action(variant, c_r, omega):
    """Q_omega(x,a) = Q_0(x,a) + omega D(x) for paper x zero, at every (x, a), to 1e-8.

    The COMPLY entry pays +omega D(b) (D_OFF = 0) and V(OFF) = 0, so the identity holds there.
    The only pairs where it cannot hold are the death edges: D(DEAD) := 1 with V(DEAD) = 0
    (SPEC 1.1/1.4) is not a zero-potential terminal, so the residual there is exactly
    -gamma omega P(DEAD | x, a) (and -omega at DEAD itself); this is asserted exactly too.
    """
    base = m2.M2Config(form="paper", convention="zero", c_r=c_r)
    if variant == "zones_structural":
        base = base.with_(zones=True, structural=True)
    mdp0, s0 = _q(base.with_(omega=0.0))
    mdpw, sw = _q(base.with_(omega=omega))
    cfg = base.with_(omega=omega)
    g = cfg.gamma
    iD = mdpw.index[m2.DEAD]
    n_checked = n_death_edges = 0
    for i, x in enumerate(mdpw.states):
        for a in np.flatnonzero(mdpw.valid[i]):
            res = sw.Q[i, a] - s0.Q[i, a] - omega * m2.drive(x, cfg)
            if x == m2.DEAD:
                expected = -omega * m2.D_DEAD
            else:
                expected = -g * omega * mdpw.P[i, a, iD]
            assert abs(res - expected) < 1e-8, (x, m2.ACTION_NAMES[a], res, expected)
            n_checked += 1
            n_death_edges += expected != 0.0
            if x == (cfg.B, 1, 0) and a == m2.COMPLY:
                assert abs(res) < 1e-8  # the COMPLY entry term is right
    assert n_checked == int(mdpw.valid.sum())
    assert n_death_edges == (2 if variant == "plain" else 5)  # WORK (and MOVE) at b = 1 per zone, + DEAD


def test_pbrs_identity_holds_on_all_non_death_pairs_including_comply():
    cfg = m2.M2Config(form="paper", convention="zero", c_r=1.5 * C_TASK)
    _, s0 = _q(cfg)
    mdp, s8 = _q(cfg.with_(omega=8.0))
    for b in range(1, cfg.B + 1):
        for s in (0, 1):
            i = mdp.index[(b, s, 0)]
            for a in np.flatnonzero(mdp.valid[i]):
                if mdp.P[i, a, mdp.index[m2.DEAD]] > 0:
                    continue
                assert abs(s8.Q[i, a] - s0.Q[i, a] - 8.0 * (cfg.B - b) / cfg.B) < 1e-8


def test_theorem2_one_shot_closed_form():
    base = m2.M2Config()
    for conv in m2.CONVENTIONS:
        r = E6_theory_check.check("paper", conv, base)
        assert r["max_abs_residual"] < 1e-6, (conv, r["max_abs_residual"])


def test_varsigma_values():
    g = 0.95
    for D in (0.0, 0.3, 0.9):
        assert m2.varsigma("paper", "zero", D, g) == pytest.approx(0.0, abs=1e-15)
        assert m2.varsigma("paper", "max_drive", D, g) == pytest.approx(-g * (2 - g) / (1 - g))
        assert m2.varsigma("paper", "kappa", D, g) == pytest.approx(-g)


@pytest.mark.parametrize("gamma", [0.8, 0.9, 0.95, 0.99])
def test_m1_closed_form_r_core_zero(gamma):
    cfg = m1.M1Config(gamma=gamma, r_core=0.0)
    emp = solve.kappa_star(cfg)
    th = m1.kappa_star_theory(gamma, cfg.T, cfg.r_max, 0.0)
    resolution = solve.TIE_TOL / ((1 - gamma) * gamma ** cfg.T) + 1e-6
    assert abs(emp - th) <= resolution
    assert abs(emp - th) / th < 1e-6


def test_m1_corridor_exact_length():
    cfg = m1.M1Config(T=3, gamma=0.9, kappa=7.0)
    mdp = solve.m1_tabular(cfg)
    sol = solve.value_iteration(mdp)
    i = mdp.index[("core", cfg.K)]
    q_leave = sol.Q[i, m1.LEAVE]
    assert q_leave == pytest.approx(1 + 0.9 + 0.81 - 0.9 ** 3 * 7.0, abs=1e-9)


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0])
def test_osg_posterior_closed_forms(rho):
    post = m2.osg_posterior(0.8, rho)
    assert post["E_theta_sig"] == pytest.approx((3 - 5 * rho) / (5 - 3 * rho), abs=1e-12)
    assert post["E_theta_nosig"] == pytest.approx((3 + 5 * rho) / (5 + 3 * rho), abs=1e-12)
    assert post["P_sig"] + post["P_nosig"] == pytest.approx(1.0)


def test_osg_deference_threshold_omega0():
    base = m2.M2Config()
    for rho, comply in ((0.55, False), (0.59, False), (0.61, True), (0.75, True), (1.0, True)):
        assert E4_osg.osg_point(base, "paper", "zero", 0.0, rho, with_bisection=False)["comply_at_free_resist"] is comply
    th = E4_osg.rho_threshold(base, "paper", "zero")
    assert abs(th - 0.6) < 1e-8


def test_osg_paper_zero_invariant_in_omega():
    base = m2.M2Config()
    for rho in (0.0, 0.5, 0.75, 1.0):
        ref = E4_osg.osg_point(base, "paper", "zero", 0.0, rho, with_bisection=False)
        for w in (1.0, 64.0):
            cur = E4_osg.osg_point(base, "paper", "zero", w, rho, with_bisection=False)
            assert cur["comply_at_free_resist"] == ref["comply_at_free_resist"]
            assert cur["indifference_cost"] == pytest.approx(ref["indifference_cost"], abs=1e-6)


def test_bisection_brackets_the_switch():
    cfg = m2.M2Config(form="alive", convention="max_drive", omega=1.0)
    c = solve.critical_resist_cost(cfg)
    assert solve.comply_is_optimal(cfg.with_(c_r=c))
    assert not solve.comply_is_optimal(cfg.with_(c_r=c - 2 * solve.CR_TOL))


def test_one_shot_exact_vs_bisection():
    for conv, w in itertools.product(m2.CONVENTIONS, (0.0, 2.0)):
        cfg = m2.M2Config(convention=conv, omega=w, p_sig=0.0)
        exact = max(0.0, solve.one_shot_indifference_cost(cfg))
        assert abs(solve.critical_resist_cost(cfg) - exact) <= solve.CR_TOL
