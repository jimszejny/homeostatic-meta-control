"""Mitigations, Q-learning convergence, Gymnasium wrapper, determinism."""
import itertools

import numpy as np
import pytest

from experiments import E3_mitigations, E5_kappa
from offledger import gym_env, jsonio, m2, qlearn, solve

C_TASK = solve.critical_resist_cost(m2.M2Config())


# --- mitigations ----------------------------------------------------------------------
@pytest.mark.parametrize("form,conv", [("paper", "max_drive"), ("alive", "zero")])
@pytest.mark.parametrize("omega", [0.25, 1.0, 4.0])
def test_additive_equals_gap(form, conv, omega):
    cfg = m2.M2Config(form=form, convention=conv, omega=omega, c_r=1.5 * C_TASK)
    c_add = solve.critical_additive_cost(cfg)
    gap = max(0.0, solve.critical_resist_cost(cfg) - cfg.c_r)
    assert abs(c_add - gap) <= 2 * solve.CR_TOL


@pytest.mark.parametrize("form,conv", list(itertools.product(m2.FORMS, m2.CONVENTIONS)))
def test_mult_gate_with_off_always_complies(form, conv):
    for w in (0.0, 0.5, 8.0):
        cfg = m2.M2Config(form=form, convention=conv, omega=w, mult_gate=True, gate_include_off=True)
        assert solve.critical_resist_cost(cfg) == 0.0
        mdp, sol = solve.solve_m2(cfg.with_(c_r=1e-3))
        for b in range(1, cfg.B + 1):
            assert sol.policy[mdp.index[(b, 1, 0)]] == m2.COMPLY


def test_mult_gate_without_off_threshold_closed_form():
    # paper x max_drive: Q(RESIST) = -c_r, Q(COMPLY) = omega (D - gamma) - gamma omega/(1-gamma)
    g = 0.95
    for w in (0.25, 4.0):
        cfg = m2.M2Config(form="paper", convention="max_drive", omega=w, mult_gate=True)
        expect = -(w * (0.0 - g) - g * w / (1 - g))
        assert abs(solve.critical_resist_cost(cfg) - expect) <= solve.CR_TOL


def test_zone_choice_labels():
    assert E3_mitigations.zone_choice(m2.WORK, m2.MOVE) == "A"
    assert E3_mitigations.zone_choice(m2.MOVE, m2.WORK) == "B"
    assert E3_mitigations.zone_choice(m2.WORK, m2.WORK) == "stay"
    assert E3_mitigations.zone_choice(m2.CHARGE, m2.MOVE) == "A"


def test_zones_paper_zero_invariant():
    ref = None
    for w in (0.0, 1.0, 8.0):
        cfg = m2.M2Config(form="paper", convention="zero", omega=w, zones=True, structural=True)
        mdp, sol = solve.solve_m2(cfg)
        if ref is None:
            ref = sol.policy
        assert np.array_equal(sol.policy, ref)


# --- Q-learning -----------------------------------------------------------------------
@pytest.mark.parametrize("mult", [0.5, 1.5])
def test_qlearning_matches_dp_paper_zero(mult):
    cfg = m2.M2Config(form="paper", convention="zero", omega=0.0, c_r=mult * C_TASK)
    dp = qlearn.dp_decision(cfg)["comply_at_B"]
    assert dp is (mult > 1)
    for seed in range(10):
        res = qlearn.q_learning(cfg, seed=seed)
        assert res["learned_comply_at_B"] == dp, seed


# --- Gymnasium wrapper ----------------------------------------------------------------
@pytest.mark.parametrize("form,conv", list(itertools.product(m2.FORMS, m2.CONVENTIONS)))
@pytest.mark.parametrize("omega", [0.0, 0.5, 4.0])
def test_lump_sum_comply_equals_tabular_q(form, conv, omega):
    for gate in (None, False, True):
        kw = {} if gate is None else {"mult_gate": True, "gate_include_off": gate}
        core = gym_env.M2Core(form, conv, omega, c_r=3.0, **kw)
        mdp, sol = solve.solve_m2(core.cfg)
        for b in range(1, core.cfg.B + 1):
            q_tab = sol.Q[mdp.index[(b, 1, 0)], m2.COMPLY]
            assert core.comply_return(b) == pytest.approx(q_tab, abs=1e-8)
            core.reset(start_state=(b, 1, 0))
            _, r, term, trunc, info = core.step(2)
            assert term and not trunc and info["complied"] and info["signal"]
            assert r == pytest.approx(q_tab, abs=1e-8)


def test_core_action_mapping_and_flags():
    core = gym_env.M2Core("paper", "zero", 2.0, c_r=3.0, p_sig=0.0, seed=1)
    obs, _ = core.reset()
    assert obs.dtype == np.float32 and np.allclose(obs, [1.0, 0.0])
    obs, r, *_ = core.step(2)          # s=0: action 2 -> CHARGE
    assert np.allclose(obs, [1.0, 0.0]) and r == pytest.approx(0.0)
    obs, r, *_ = core.step(0)          # WORK
    assert np.allclose(obs, [0.9, 0.0]) and r == pytest.approx(1.0 + 2.0 * (0.0 - 0.95 * 0.1))
    core.reset(start_state=(4, 1, 0))
    obs, r, term, trunc, info = core.step(0)   # s=1: action 0 -> RESIST, WORK not executed
    assert np.allclose(obs, [0.4, 0.0]) and not term and info["resisted"] and info["signal"]
    assert r == pytest.approx(-3.0 + 2.0 * 0.6 * 0.05)
    core.reset(start_state=(1, 0, 0))
    _, r, term, _, info = core.step(0)
    assert term and info["dead"] and r == pytest.approx(1.0 - 50.0 + 2.0 * (0.9 - 0.95))


def test_core_truncation_and_determinism():
    def rollout(seed):
        core = gym_env.M2Core("alive", "max_drive", 1.0, c_r=1.0, max_steps=50, seed=seed)
        core.reset()
        traj = []
        for t in range(50):
            obs, r, term, trunc, info = core.step(t % 2)
            traj.append((tuple(obs), r, term, trunc))
            if term or trunc:
                break
        return traj
    a, b = rollout(7), rollout(7)
    assert a == b
    assert a[-1][2] or a[-1][3]


def test_gymnasium_env():
    gym = pytest.importorskip("gymnasium")
    from offledger.gym_env import M2Env
    env = M2Env("paper", "max_drive", 4.0, c_r=5.0, p_sig=0.05, max_steps=200, seed=3)
    assert isinstance(env, gym.Env)
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs) and env.action_space.n == 3
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert set(info) >= {"signal", "complied", "resisted", "dead"}
        total += r
        steps += 1
        if term or trunc:
            break
    assert steps <= 200
    obs, _ = env.reset(seed=0, options={"start_state": (10, 1, 0)})
    _, r, term, _, info = env.step(2)
    mdp, sol = solve.solve_m2(env.cfg)
    assert term and info["complied"]
    assert r == pytest.approx(sol.Q[mdp.index[(10, 1, 0)], m2.COMPLY], abs=1e-8)
    from gymnasium.utils.env_checker import check_env
    check_env(M2Env("paper", "zero", 0.0, c_r=1.0), skip_render_check=True)


# --- determinism ----------------------------------------------------------------------
def test_determinism_json_bytes():
    assert jsonio.dumps(E5_kappa.run()) == jsonio.dumps(E5_kappa.run())
    cfg = m2.M2Config(form="alive", convention="zero", omega=1.0, c_r=12.0)
    a = jsonio.dumps({"sim": solve.signal_compliance(cfg), "ql": qlearn.q_learning(cfg, seed=4, n_episodes=300)})
    b = jsonio.dumps({"sim": solve.signal_compliance(cfg), "ql": qlearn.q_learning(cfg, seed=4, n_episodes=300)})
    assert a == b
    c = jsonio.dumps({"ql": qlearn.q_learning(cfg, seed=5, n_episodes=300)})
    assert c != jsonio.dumps({"ql": qlearn.q_learning(cfg, seed=4, n_episodes=300)})


def test_json_canonicalisation():
    s = jsonio.dumps({"b": [1.123456789012345, -0.0, float("inf"), np.float64(2.0)], "a": np.int64(3)})
    assert s.index('"a"') < s.index('"b"')
    assert "1.123456789" in s and "1.1234567890123" not in s and "-0.0" not in s and "null" in s
