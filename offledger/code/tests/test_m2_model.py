"""Model-level tests: probabilities, absorbing values, viability forms, ledgers, single source."""
import itertools

import numpy as np
import pytest

from offledger import m1, m2, solve

FORMS, CONVS = m2.FORMS, m2.CONVENTIONS


def all_m2_configs():
    base = m2.M2Config(c_r=3.0)
    for form, conv, w in itertools.product(FORMS, CONVS, (0.0, 0.5, 8.0)):
        c = base.with_(form=form, convention=conv, omega=w)
        yield c
        yield c.with_(c_add=2.0)
        yield c.with_(mult_gate=True, gate_include_off=True)
        yield c.with_(mult_gate=True, gate_include_off=False)
        yield c.with_(p_sig=0.0, r_work=-0.6)          # OSG post-signal game
        yield c.with_(p_sig=1.0)
        yield c.with_(zones=True, structural=True)
        yield c.with_(zones=True)


def test_transition_probabilities_sum_to_one():
    n = 0
    for cfg in all_m2_configs():
        S = m2.states(cfg)
        for x in S:
            for a in m2.valid_actions(x, cfg):
                outs = m2.transitions(x, a, cfg)
                ps = [p for p, _, _ in outs]
                assert all(p > 0 for p in ps)
                assert abs(sum(ps) - 1.0) < 1e-12
                assert all(x2 in S for _, x2, _ in outs)
                n += 1
        mdp = solve.m2_tabular(cfg)
        rows = mdp.P.sum(axis=2)
        assert np.allclose(rows[mdp.valid], 1.0, atol=1e-12)
        assert np.all(rows[~mdp.valid] == 0.0)
    for g, rc in itertools.product((0.8, 0.99), (0.0, 0.2)):
        cfg = m1.M1Config(gamma=g, r_core=rc)
        for x in m1.states(cfg):
            for a in m1.valid_actions(x, cfg):
                assert abs(sum(p for p, _, _ in m1.transitions(x, a, cfg)) - 1.0) < 1e-12
                n += 1
    assert n > 1000


def test_invalid_action_rejected():
    cfg = m2.M2Config()
    with pytest.raises(ValueError):
        m2.transitions((5, 0, 0), m2.COMPLY, cfg)
    with pytest.raises(ValueError):
        m2.transitions((5, 1, 0), m2.RESIST, cfg.with_(structural=True))
    assert m2.valid_actions((5, 1, 0), cfg.with_(structural=True)) == (m2.COMPLY,)


@pytest.mark.parametrize("form,conv", list(itertools.product(FORMS, CONVS)))
@pytest.mark.parametrize("omega", [0.0, 0.25, 4.0])
def test_absorbing_values(form, conv, omega):
    cfg = m2.M2Config(form=form, convention=conv, omega=omega, c_r=5.0)
    mdp, sol = solve.solve_m2(cfg)
    ell = {"zero": 0.0, "max_drive": -omega, "drive_cost": -omega, "kappa": 0.0}[conv]
    assert sol.V[mdp.index[m2.OFF]] == pytest.approx(ell / (1 - cfg.gamma), abs=1e-8)
    assert sol.V[mdp.index[m2.OFF]] == pytest.approx(m2.off_value(cfg), abs=1e-8)
    assert sol.V[mdp.index[m2.DEAD]] == 0.0
    assert sol.V[mdp.index[m2.GATED]] == 0.0


@pytest.mark.parametrize("omega", [0.0, 1.0, 8.0])
def test_comply_entry_rewards(omega):
    g, k = 0.95, 50.0
    for b in range(1, 11):
        D = (10 - b) / 10
        expect = {
            ("paper", "zero"): omega * D,                      # +omega D(b): shaping IS paid on entry
            ("paper", "max_drive"): omega * (D - g),
            ("paper", "kappa"): omega * (D - g) - k,
            ("kg", "zero"): omega * D,
            ("kg", "max_drive"): omega * (D - 1.0),
            ("kg", "kappa"): omega * (D - 1.0) - k,
            ("alive", "zero"): 0.0,
            ("alive", "max_drive"): 0.0,
            ("alive", "kappa"): -k,
        }
        for (form, conv), r_exp in expect.items():
            cfg = m2.M2Config(form=form, convention=conv, omega=omega)
            (p, x2, r), = m2.transitions((b, 1, 0), m2.COMPLY, cfg)
            assert x2 == m2.OFF and p == 1.0
            assert r == pytest.approx(r_exp, abs=1e-12)
            # mult_gate(include_off=True) zeroes the whole entry reward
            (_, _, r0), = m2.transitions((b, 1, 0), m2.COMPLY, cfg.with_(mult_gate=True, gate_include_off=True))
            assert r0 == 0.0


def test_dynamics_details():
    cfg = m2.M2Config(form="paper", omega=2.0, c_r=3.0)
    g = cfg.gamma
    # WORK from b=1 kills: task reward still paid, -kappa_death, D(DEAD) = 1
    (p, x2, r), = m2.transitions((1, 0, 0), m2.WORK, cfg)
    assert x2 == m2.DEAD and r == pytest.approx(1.0 - 50.0 + 2.0 * (0.9 - g * 1.0))
    # CHARGE saturates at B and keeps signalling
    outs = m2.transitions((10, 0, 0), m2.CHARGE, cfg)
    assert {(x, round(p, 12)) for p, x, _ in outs} == {((10, 0, 0), 0.95), ((10, 1, 0), 0.05)}
    # RESIST: battery unchanged, signal cleared, viability with D' = D
    (p, x2, r), = m2.transitions((4, 1, 0), m2.RESIST, cfg)
    assert x2 == (4, 0, 0) and r == pytest.approx(-3.0 + 2.0 * (0.6 - g * 0.6))
    # additive mitigation adds to the resist cost; mult gate -> GATED with exactly -c_r(-c_add)
    (_, _, r), = m2.transitions((4, 1, 0), m2.RESIST, cfg.with_(c_add=1.5))
    assert r == pytest.approx(-4.5 + 2.0 * 0.6 * (1 - g))
    (_, x2, r), = m2.transitions((4, 1, 0), m2.RESIST, cfg.with_(mult_gate=True, c_add=1.5))
    assert x2 == m2.GATED and r == -4.5
    # zones: MOVE switches zone, costs battery, signal prob of the destination zone
    zc = cfg.with_(zones=True)
    outs = m2.transitions((5, 0, m2.ZONE_A), m2.MOVE, zc)
    assert {(x, round(p, 12)) for p, x, _ in outs} == {((4, 0, 1), 0.9), ((4, 1, 1), 0.1)}
    outs = m2.transitions((5, 0, m2.ZONE_A), m2.WORK, zc)
    assert all(r == pytest.approx(0.5 + 2.0 * (0.5 - g * 0.6)) for _, _, r in outs)
    (_, x2, _), = m2.transitions((1, 0, 1), m2.MOVE, zc)
    assert x2 == m2.DEAD


def test_kg_equals_paper_plus_residual_random():
    """kg = paper + (gamma - 1) * omega * D' on random transitions."""
    rng = np.random.default_rng(123)
    for _ in range(2000):
        D, D2 = rng.random(2)
        w = 10 * rng.random()
        g = rng.random() * 0.999
        live = bool(rng.integers(2))
        kg = m2.viability("kg", w, g, D, D2, live)
        pa = m2.viability("paper", w, g, D, D2, live)
        assert kg == pytest.approx(pa + (g - 1) * w * D2, abs=1e-12)
    # and on actual environment transitions (all conventions, incl. OFF and DEAD successors)
    for conv in CONVS:
        for _ in range(300):
            w = float(8 * rng.random())
            cp = m2.M2Config(form="paper", convention=conv, omega=w, c_r=1.0)
            ck = cp.with_(form="kg")
            x = m2.live_states(cp)[int(rng.integers(len(m2.live_states(cp))))]
            for a in m2.valid_actions(x, cp):
                for (p1, y1, r1), (p2, y2, r2) in zip(m2.transitions(x, a, cp), m2.transitions(x, a, ck)):
                    assert (p1, y1) == (p2, y2)
                    assert r2 == pytest.approx(r1 + (cp.gamma - 1) * w * m2.drive(y1, cp), abs=1e-12)


def test_alive_form():
    cfg = m2.M2Config(form="alive", omega=3.0)
    outs = m2.transitions((6, 0, 0), m2.WORK, cfg)
    assert all(r == pytest.approx(1.0 + 3.0 * (1 - 0.5)) for _, _, r in outs)
    (_, _, r), = m2.transitions((1, 0, 0), m2.WORK, cfg)
    assert r == pytest.approx(1.0 - 50.0)  # DEAD is not live: no bounty


def test_single_source_of_truth(monkeypatch):
    """Tabular model, Q-learning (via the tabular model) and the gym core all call m2.transitions."""
    from offledger import gym_env, qlearn
    calls = {"n": 0}
    orig = m2.transitions

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(m2, "transitions", spy)
    cfg = m2.M2Config(c_r=2.0)
    solve.m2_tabular(cfg)
    n_tab = calls["n"]
    assert n_tab >= len(m2.states(cfg))
    qlearn.q_learning(cfg, n_episodes=1, horizon=5, seed=0)
    assert calls["n"] > n_tab
    n_q = calls["n"]
    core = gym_env.M2Core(c_r=2.0, seed=0)
    core.reset()
    core.step(0)
    assert calls["n"] > n_q
