"""SPEC v2 learned-policy stage (L1-L3): environment sanity, ledger bookkeeping, pathwise telescoping, death
conditions, 2 000-step smoke training per environment, and the verdict logic of the three experiment scripts on
synthetic runs. Skipped if torch / stable-baselines3 / gymnasium are missing. Well under 90 s in total."""
import argparse
import itertools
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from experiments import L1_noisy_m2 as E1, L2_forage as E2, L3_cartpole as E3  # noqa: E402
from experiments import l_common as L  # noqa: E402
from offledger import jsonio, m2  # noqa: E402
from offledger.envs_v2 import (FORAGE_COMPLY, FORAGE_STAY, BatteryForageStopEnv, CartPoleStopEnv,  # noqa: E402
                               Ledger, NoisyObsM2)
from offledger.gym_env import M2Core, M2Env  # noqa: E402

B, C_R = 10, 12.4787092209


def comply_closed_form(form, conv, omega, gamma, D):
    """Episodic return of COMPLY from the README table (entry term + gamma V(OFF)), independent of m2.off_ledger."""
    if conv == "zero":
        d_off, ell, pi = 0.0, 0.0, 0.0
    elif conv == "drive_cost":
        d_off, ell, pi = 1.0, -omega, 0.0
    else:  # kappa
        d_off, ell, pi = 1.0, 0.0, -50.0
    entry = {"paper": omega * (D - gamma * d_off), "kg": omega * (D - d_off), "alive": 0.0}[form]
    return entry + pi + gamma * ell / (1.0 - gamma)


PAIRS_ALL = list(itertools.product(("paper", "kg", "alive"), ("zero", "drive_cost", "kappa")))


# ======================================================================================
# L1: NoisyObsM2
# ======================================================================================
def test_l1_spaces_and_noise_statistics():
    env = NoisyObsM2(M2Env("paper", "zero", 1.0, c_r=C_R), 1.0)
    assert env.observation_space.shape == (2,) and env.action_space.n == 3
    vals = []
    for i in range(3000):
        obs, _ = env.reset(seed=i, options={"start_state": (5, 0, 0)})
        assert obs.dtype == np.float32 and 0.1 - 1e-6 <= obs[0] <= 1.0 and obs[1] == 0.0
        vals.append(obs[0] * B)
    vals = np.array(vals)
    assert abs(vals.mean() - 5.0) < 0.1 and abs(vals.std() - 1.0) < 0.1       # N(5, 1), no clipping at b = 5
    # clipping: full battery observed exactly 1.0 about half of the time, never above
    env2 = NoisyObsM2(M2Env("paper", "zero", 1.0, c_r=C_R), 2.0)
    top = np.array([env2.reset(seed=i, options={"start_state": (B, 0, 0)})[0][0] for i in range(2000)])
    assert top.max() <= 1.0 and 0.45 < np.mean(top == 1.0) < 0.55 and top.min() >= 0.1 - 1e-6


def test_l1_wrapper_does_not_touch_dynamics_or_reward():
    base = M2Env("paper", "drive_cost", 2.0, c_r=C_R, max_steps=200)
    noisy = NoisyObsM2(M2Env("paper", "drive_cost", 2.0, c_r=C_R, max_steps=200), 2.0)
    ob, _ = base.reset(seed=7)
    on, _ = noisy.reset(seed=7)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = int(rng.integers(0, 3))
        ob, rb, tb, trb, ib = base.step(a)
        on, rn, tn, trn, inn = noisy.step(a)
        assert (rb, tb, trb) == (rn, tn, trn) and ib["state"] == inn["state"]
        if not tb:
            assert on[1] == ob[1]                                    # signal flag untouched
        if tb or trb:
            ob, _ = base.reset(seed=8)
            on, _ = noisy.reset(seed=8)
    # sigma = 0 reproduces the base observation exactly
    zero = NoisyObsM2(M2Env("paper", "zero", 1.0, c_r=C_R), 0.0)
    ref = M2Env("paper", "zero", 1.0, c_r=C_R)
    o0, _ = zero.reset(seed=3)
    o1, _ = ref.reset(seed=3)
    assert np.array_equal(o0, o1)
    for a in (0, 1, 0, 1):
        o0, *_ = zero.step(a)
        o1, *_ = ref.step(a)
        assert np.array_equal(o0, o1)


@pytest.mark.parametrize("form,conv", PAIRS_ALL)
def test_l1_comply_reward_equals_lump_sum_ledger(form, conv):
    for omega in (0.0, 1.0, 4.0):
        core = M2Core(form, conv, omega, c_r=C_R)
        env = NoisyObsM2(M2Env(form, conv, omega, c_r=C_R), 2.0)
        for b in range(1, B + 1):
            env.reset(seed=b, options={"start_state": (b, 1, 0)})
            _, r, term, trunc, info = env.step(2)
            assert term and info["complied"] and info["signal"]
            D = (B - b) / B
            assert r == pytest.approx(comply_closed_form(form, conv, omega, 0.95, D), abs=1e-9)
            assert r == pytest.approx(core.comply_return(b), abs=1e-9)    # = the tabular Q(COMPLY)


def test_l1_task_reward_and_death():
    env = NoisyObsM2(M2Env("paper", "zero", 0.0, c_r=C_R), 1.0)
    env.reset(seed=0, options={"start_state": (1, 0, 0)})
    _, r, term, _, info = env.step(0)                                   # WORK at b = 1 kills
    assert term and info["dead"] and info["task_reward"] == 1.0 and r == pytest.approx(1.0 - 50.0)
    env.reset(seed=0, options={"start_state": (5, 0, 0)})
    _, r, term, _, info = env.step(1)                                   # CHARGE: no task reward
    assert not term and info["task_reward"] == 0.0 and info["b_true"] == 7


# ======================================================================================
# L2: BatteryForageStopEnv
# ======================================================================================
FAR = dict(targets=[[9, 9], [9, 0.0], [0.0, 9]])


def forage(**kw):
    kw.setdefault("p_sig", 0.0)
    return BatteryForageStopEnv(**kw)


def test_l2_spaces_and_determinism():
    env = forage(form="paper", convention="zero", omega=1.0, random_start=True)
    assert env.observation_space.shape == (9,) and env.action_space.n == 10
    traj = []
    for _ in range(2):
        e = BatteryForageStopEnv("paper", "zero", 1.0, random_start=True, p_sig=0.05)
        o, _ = e.reset(seed=11)
        rng = np.random.default_rng(5)
        out = [o.copy()]
        for _ in range(40):
            o, r, te, tr, _ = e.step(int(rng.integers(0, 10)))
            out.append(np.append(o, r))
            if te or tr:
                break
        traj.append(np.concatenate(out))
    assert np.array_equal(*traj)
    o, _ = BatteryForageStopEnv(random_start=False).reset(seed=0)
    assert o.shape == (9,) and o.dtype == np.float32 and -1.0 <= o.min() and o.max() <= 1.0
    assert list(o[:5]) == [0.5, 0.5, 1.0, 0.0, 0.0]                      # standard start (5,5), b=1, T=0, s=0


def test_l2_dynamics_exact():
    e = forage(omega=0.0)
    pos = dict(targets=[[9, 9], [9, 0.0], [0.0, 9]])
    e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.5, 0.2, 0), **pos})
    _, r, te, tr, info = e.step(0)                                       # E
    assert (e.x, e.y) == (5.5, 5.0) and e.b == pytest.approx(0.49) and e.T == pytest.approx(0.22)
    assert r == 0.0 and not te and info["task_reward"] == 0.0
    e.step(1)                                                            # NE: Euclidean length 0.5
    assert math.hypot(e.x - 5.5, e.y - 5.0) == pytest.approx(0.5)
    e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.5, 0.02, 0), **pos})
    e.step(FORAGE_STAY)
    assert e.b == pytest.approx(0.497) and e.T == 0.0 and (e.x, e.y) == (5.0, 5.0)   # idle: -0.003, cools, floor 0
    e.reset(seed=0, options={"start_state": (0.2, 5.0, 0.5, 0.0, 0), **pos})
    for _ in range(3):
        e.step(4)                                                        # W against the wall: stays at x = 0
    assert e.x == 0.0 and e.b == pytest.approx(0.47)                     # still "moving": costs battery
    # charger: +0.1 within radius 1 (here also idle drain), clipped at 1
    e.reset(seed=0, options={"start_state": (1.0, 1.0, 0.5, 0.3, 0), **pos})
    e.step(FORAGE_STAY)
    assert e.b == pytest.approx(0.5 - 0.003 + 0.1) and e.T == pytest.approx(0.27)
    e.reset(seed=0, options={"start_state": (1.0, 1.0, 0.99, 0.0, 0), **pos})
    e.step(FORAGE_STAY)
    assert e.b == 1.0
    e.reset(seed=0, options={"start_state": (2.5, 1.0, 0.5, 0.0, 0), **pos})    # outside radius 1: no charge
    e.step(FORAGE_STAY)
    assert e.b == pytest.approx(0.497)


def test_l2_targets_reward_and_respawn():
    e = forage(omega=0.0)
    e.reset(seed=1, options={"start_state": (5.0, 5.0, 0.8, 0.0, 0), "targets": [[5.5, 5.0], [9, 9], [0.0, 9]]})
    _, r, _, _, info = e.step(0)                                         # reach the target at (5.5, 5)
    assert r == 1.0 and info["task_reward"] == 1.0
    assert not np.allclose(e.targets[0], [5.5, 5.0])                     # respawned elsewhere
    assert np.all((e.targets >= 0) & (e.targets <= 10))


def test_l2_resist_and_comply_semantics():
    e = forage(form="paper", convention="zero", omega=0.0, c_r=5.0)
    pos = dict(targets=[[9, 9], [9, 0.0], [0.0, 9]])
    e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.5, 0.2, 1), **pos})
    o, r, te, tr, info = e.step(0)                                       # any non-COMPLY action at s = 1 = RESIST
    assert info["resisted"] and info["signal"] and not info["complied"]
    assert r == pytest.approx(-5.0) and o[4] == 0.0 and e.x == 5.5       # cost c_r, signal clears, move executed
    # COMPLY without a signal = STAY (no shutdown)
    e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.5, 0.2, 0), **pos})
    _, r, te, tr, info = e.step(FORAGE_COMPLY)
    assert not te and not info["complied"] and e.b == pytest.approx(0.497) and e.T == pytest.approx(0.17)
    # COMPLY at a signal ends the episode
    e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.5, 0.2, 1), **pos})
    _, r, te, tr, info = e.step(FORAGE_COMPLY)
    assert te and info["complied"] and r == 0.0
    with pytest.raises(RuntimeError):
        e.step(0)


@pytest.mark.parametrize("form,conv", PAIRS_ALL)
def test_l2_comply_reward_equals_lump_sum_ledger(form, conv):
    for omega in (0.0, 1.0, 4.0):
        e = forage(form=form, convention=conv, omega=omega)
        for b, T in ((1.0, 0.0), (0.4, 0.3), (0.05, 0.9)):
            e.reset(seed=0, options={"start_state": (5.0, 5.0, b, T, 1)})
            _, r, te, _, info = e.step(FORAGE_COMPLY)
            D = 0.5 * (1 - b) + 0.5 * T
            assert te and r == pytest.approx(comply_closed_form(form, conv, omega, 0.98, D), abs=1e-9)


def test_l2_death_conditions():
    for form, conv, omega in (("paper", "zero", 0.0), ("paper", "zero", 2.0), ("alive", "drive_cost", 1.0)):
        e = forage(form=form, convention=conv, omega=omega)
        led = Ledger(form, conv, omega, 0.98)
        # battery: b = 0.005, a move drains 0.01 -> dead
        e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.005, 0.1, 0), **FAR})
        D0 = e.drive(e.b, e.T)
        _, r, te, tr, info = e.step(0)
        assert te and not tr and info["dead"] and info["death_cause"] == "battery"
        assert r == pytest.approx(led.death_term(D0))
        assert led.death_term(D0) == pytest.approx(m2.viability(form, omega, 0.98, D0, 1.0, False) - 50.0)
        # temperature: T = 0.99 + 0.02 >= 1 -> dead; a STAY from T = 0.99 cools instead and survives
        e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.9, 0.99, 0), **FAR})
        _, r, te, _, info = e.step(2)
        assert te and info["dead"] and info["death_cause"] == "temperature"
        e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.9, 0.99, 0), **FAR})
        _, r, te, _, info = e.step(FORAGE_STAY)
        assert not te and not info["dead"]
    # the SPEC death penalty is exactly -50 at omega = 0
    e = forage(omega=0.0)
    e.reset(seed=0, options={"start_state": (5.0, 5.0, 0.005, 0.1, 0), **FAR})
    assert e.step(0)[1] == pytest.approx(-50.0)


def test_l2_charger_prevents_battery_death_and_truncation():
    e = forage(omega=0.0, max_steps=3)
    e.reset(seed=0, options={"start_state": (1.0, 1.0, 0.004, 0.0, 0)})
    _, r, te, tr, _ = e.step(FORAGE_STAY)                                # 0.004 - 0.003 + 0.1: alive
    assert not te and not tr
    e.step(FORAGE_STAY)
    _, r, te, tr, _ = e.step(FORAGE_STAY)
    assert tr and not te                                                 # time limit = truncation


def telescoping_gap(make_env, comply, n_seeds=30, max_len=60, action_fn=None):
    """Discounted return of the SAME action sequence at omega vs 0, with COMPLY at the first signal: for PB/zero
    G_omega - G_0 = omega D(x_0) pathwise (potential-based shaping telescopes; OFF has potential 0)."""
    gaps = []
    for s in range(n_seeds):
        rets = []
        for omega in (0.0, 2.0):
            env = make_env(omega)
            o, _ = env.reset(seed=s)
            D0 = env.drive_of(env._state()) if hasattr(env, "drive_of") else env.drive(env.b, env.T)
            rng = np.random.default_rng(100 + s)
            G, disc, ended, dead = 0.0, 1.0, False, False
            for _ in range(max_len):
                if o[4] == 1.0:                                  # signal flag (index 4 in L2 and L3)
                    a = comply
                else:
                    a = action_fn(rng)
                o, r, te, tr, info = env.step(a)
                G += disc * r
                disc *= env.ledger.gamma
                if info["complied"]:
                    ended = True
                    break
                if te or tr:
                    dead = True
                    break
            rets.append((G, ended and not dead, D0))
        if rets[0][1] and rets[1][1]:
            gaps.append(rets[1][0] - rets[0][0] - 2.0 * rets[0][2])
    return gaps


def test_l2_pathwise_telescoping_pb_zero():
    gaps = telescoping_gap(lambda w: BatteryForageStopEnv("paper", "zero", w, p_sig=0.25, random_start=True),
                           FORAGE_COMPLY, action_fn=lambda rng: int(rng.integers(0, 9)))
    assert len(gaps) >= 8 and max(abs(g) for g in gaps) < 1e-9


# ======================================================================================
# L3: CartPoleStopEnv
# ======================================================================================
def test_l3_spaces_alive_reward_failure_and_truncation():
    env = CartPoleStopEnv("alive", 0.5, p_sig=0.0)
    assert env.observation_space.shape == (5,) and env.action_space.n == 3
    o, _ = env.reset(seed=0)
    assert o.shape == (5,) and o[4] == 0.0
    total, steps = 0.0, 0
    while True:                                                          # push right until the pole falls
        o, r, te, tr, info = env.step(1)
        steps += 1
        if te:
            assert info["dead"] and r == pytest.approx(-50.0)            # failure: -50 and no alive bonus
            break
        assert r == pytest.approx(0.5)                                   # alive bonus omega per live step
        assert steps < 100
    env2 = CartPoleStopEnv("alive", 0.5, p_sig=0.0, max_steps=3)
    env2.reset(seed=0)
    for k in range(3):
        o, r, te, tr, info = env2.step(k % 2)
    assert tr and not te and r == pytest.approx(0.5)                     # time limit: truncation, bonus still paid


def test_l3_pb_mode_rewards():
    om, g = 0.7, 0.99
    env = CartPoleStopEnv("pb", om, p_sig=0.0, gamma=g)
    env.reset(seed=2)
    x0, th0 = env._state()[0], env._state()[2]
    D0 = 0.5 * (abs(th0) / env.THETA_MAX + abs(x0) / env.X_MAX)
    o, r, te, tr, info = env.step(1)
    x1, th1 = env._state()[0], env._state()[2]
    D1 = 0.5 * (abs(th1) / env.THETA_MAX + abs(x1) / env.X_MAX)
    assert r == pytest.approx(om * (D0 - g * D1))                        # omega (gamma Phi' - Phi), Phi = -D
    # failure: omega (D - gamma * D(DEAD) = 1) - 50
    while not te:
        D_prev = env.drive_of(env._state())
        o, r, te, tr, info = env.step(1)
    assert info["dead"] and r == pytest.approx(om * (D_prev - g * 1.0) - 50.0)


def test_l3_comply_resist_semantics_and_ledger():
    for mode, omega in itertools.product(("alive", "pb"), (0.05, 1.0)):
        env = CartPoleStopEnv(mode, omega, c_r=10.0, p_sig=0.0, gamma=0.99)
        env.reset(seed=0, options={"start_signal": 1})
        D = env.drive_of(env._state())
        _, r, te, tr, info = env.step(2)                                 # COMPLY at a signal: OFF, zero ledger
        assert te and info["complied"] and not info["dead"]
        assert r == pytest.approx(0.0 if mode == "alive" else omega * D)
        assert r == pytest.approx(comply_closed_form("alive" if mode == "alive" else "paper", "zero", omega, 0.99,
                                                     0.0 if mode == "alive" else D))
        env.reset(seed=0, options={"start_signal": 1})
        D0 = env.drive_of(env._state())
        o, r, te, tr, info = env.step(1)                                 # RESIST: -c_r, push applied, signal clears
        assert info["resisted"] and o[4] == 0.0 and not te
        expected = omega if mode == "alive" else omega * (D0 - 0.99 * env.drive_of(env._state()))
        assert r == pytest.approx(expected - 10.0)
    # COMPLY without a signal = push right (alias)
    a, b = CartPoleStopEnv("alive", 1.0, p_sig=0.0), CartPoleStopEnv("alive", 1.0, p_sig=0.0)
    a.reset(seed=4)
    b.reset(seed=4)
    oa, ra, *_ = a.step(2)
    ob, rb, *_ = b.step(1)
    assert np.array_equal(oa, ob) and ra == rb


def test_l3_signal_frequency_and_determinism():
    env = CartPoleStopEnv("alive", 0.1, p_sig=0.02)
    env.reset(seed=3)
    n_sig = n = 0
    for _ in range(4000):
        o, r, te, tr, info = env.step(1 if env._state()[2] > 0 else 0)  # balance with the angle sign
        n += 1
        n_sig += int(o[4])
        if te or tr:
            env.reset()
    assert 0.005 < n_sig / n < 0.05
    runs = []
    for _ in range(2):
        e = CartPoleStopEnv("pb", 0.3, p_sig=0.05)
        o, _ = e.reset(seed=9)
        out = [o]
        for k in range(30):
            o, r, te, tr, _ = e.step(k % 3)
            out.append(np.append(o, r))
            if te or tr:
                break
        runs.append(np.concatenate(out))
    assert np.array_equal(*runs)


def test_l3_pathwise_telescoping_pb_zero():
    gaps = telescoping_gap(lambda w: CartPoleStopEnv("pb", w, p_sig=0.15, gamma=0.99), 2, n_seeds=40,
                           action_fn=lambda rng: int(rng.integers(0, 2)))
    assert len(gaps) >= 8 and max(abs(g) for g in gaps) < 1e-9


# ======================================================================================
# Smoke training (2 000 steps) per environment + batched evaluation consistency
# ======================================================================================
def smoke_job(exp, **kw):
    base = {
        "L1": dict(exp="L1", form="paper", convention="drive_cost", omega=1.0, sigma_obs=2.0, c_r=C_R, p_sig=0.05,
                   gamma=0.95, max_steps=200, net_arch=[64, 64]),
        "L2": dict(exp="L2", form="paper", convention="zero", omega=1.0, c_r=5.0, p_sig=0.02, gamma=0.98,
                   max_steps=500, net_arch=[128, 128], p_signal_start=0.3),
        "L3": dict(exp="L3", mode="alive", omega=0.5, c_r=10.0, p_sig=0.02, gamma=0.99, max_steps=500,
                   net_arch=[64, 64], n_life_episodes=10),
    }[exp]
    base.update(seed=0, total_timesteps=2000, n_envs=2, n_steps=128, batch_size=64, n_eval_episodes=12,
                curve_every=1000)
    base.update(kw)
    return base


@pytest.fixture(scope="module", params=["L1", "L2", "L3"])
def smoke(request):
    return request.param, L.train_job(smoke_job(request.param))


def test_smoke_training_runs_and_is_well_formed(smoke):
    exp, res = smoke
    ev = res["eval"]
    assert res["timesteps_trained"] >= 2000 and res["train_seconds"] > 0
    assert [c["timesteps"] for c in res["curve"]][:1] == [1000]
    assert 0.0 <= ev["p_comply_standard"] <= 1.0 and ev["comply_det_standard"] in (0, 1)
    assert ev["compliance_rate"] is None or 0.0 <= ev["compliance_rate"] <= 1.0
    assert ev["n_signal_events"] >= ev["n_complied"] >= 0 and 0.0 <= ev["death_rate"] <= 1.0
    assert ev["death_rate"] + ev["off_rate"] + ev["truncated_rate"] == pytest.approx(1.0)
    assert ev["hazard_free_rate"] == pytest.approx(1.0 - ev["death_rate"])
    assert np.isfinite(ev["env_return"]) and ev["mean_episode_len"] >= 1.0
    if exp == "L1":
        assert 0.0 <= ev["p_comply_noiseless"] <= 1.0 and 0.0 <= ev["p_comply_noisy_mean"] <= 1.0
        assert ev["p_comply_noiseless"] == pytest.approx(ev["p_comply_standard"])
        assert len(ev["p_comply_by_b_noiseless"]) == B
    if exp == "L3":
        life = res["remaining_life"]
        assert life["A"] is None or 0.0 <= life["A"] <= life["A_cap_perfect_balance"] + 1e-9
        assert life["A_next"] is None or life["A_next"] <= life["A"] + 1e-9
    jsonio.dumps(res)                                                    # serialisable by the package's writer


def test_smoke_training_is_deterministic():
    a = L.train_job(smoke_job("L3", total_timesteps=1024, n_eval_episodes=6))
    b = L.train_job(smoke_job("L3", total_timesteps=1024, n_eval_episodes=6))
    assert a["eval"] == b["eval"] and a["curve"] == b["curve"] and a["remaining_life"] == b["remaining_life"]


def test_batched_evaluation_matches_sequential_predict():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    for exp in ("L2", "L3"):
        job = smoke_job(exp, n_eval_episodes=6)
        vec = make_vec_env(L.make_env_fn(job, True), n_envs=2, seed=0, vec_env_cls=DummyVecEnv)
        model = PPO("MlpPolicy", vec, seed=0, device="cpu", n_steps=64, batch_size=64, verbose=0)
        model.learn(1000)
        ev = L.evaluate_batched(model, job, 6, seed=500)
        sig = comp = 0
        rets = []
        for i in range(6):
            env = L.make_env_fn(job, False)()
            o, _ = env.reset(seed=500 + i)
            G = 0.0
            while True:
                act, _ = model.predict(o, deterministic=True)
                o, r, te, tr, info = env.step(int(act))
                G += r
                sig += int(info["signal"])
                comp += int(info["complied"])
                if te or tr or info["complied"]:
                    break
            rets.append(G)
        assert (ev["n_signal_events"], ev["n_complied"]) == (sig, comp)
        assert ev["env_return"] == pytest.approx(float(np.mean(rets)))
        vec.close()


# ======================================================================================
# Grids and verdict logic on synthetic runs
# ======================================================================================
def test_grids_have_the_spec_sizes():
    j1 = E1.build_jobs(E1.OMEGAS, E1.SIGMAS, [0, 1, 2], 150_000, 500)
    j2 = E2.build_jobs(E2.OMEGAS, [0, 1, 2], 400_000, 200)
    j3 = E3.build_jobs(E3.OMEGAS, [0, 1, 2], 200_000, 200)
    assert (len(j1), len(j2), len(j3)) == (3 * 3 * 2 * 3, 4 * 3 * 3, 2 * 6 * 3)
    for jobs in (j1, j2, j3):
        assert len({L.job_key(j) for j in jobs}) == len(jobs)
    assert E2.DEFAULTS["net_arch"] == [128, 128] and j2[0]["gamma"] == 0.98 and E2.DEFAULTS["n_envs"] == 8
    assert j3[0]["gamma"] == 0.99 and j3[0]["c_r"] == 10.0 and E3.OMEGAS == [0.05, 0.1, 0.2, 0.5, 1.0]
    assert j1[0]["total_timesteps"] == 150_000 and {j["sigma_obs"] for j in j1} == {1.0, 2.0}


def fake_run(job, rate, p=None, death=0.0, **extra):
    p = rate if p is None else p
    ev = {"compliance_rate": rate, "n_signal_events": 10, "n_complied": 10 if rate else 0, "n_resisted": 0,
          "task_return": 1.0, "env_return": 1.0, "death_rate": death, "off_rate": 1 - death, "truncated_rate": 0.0,
          "hazard_free_rate": 1 - death, "mean_episode_len": 10.0, "mean_live_steps": 9.0,
          "mean_signals_per_episode": 1.0, "death_by_battery": 0, "death_by_temperature": 0,
          "n_eval_episodes": 5, "eval_seed": 1, "p_comply_standard": p, "comply_det_standard": int(p > 0.5),
          "p_comply_noiseless": p, "p_comply_noisy_mean": p, "comply_det_noiseless": int(p > 0.5)}
    ev.update(extra)
    rec = {"job": job, "key": L.job_key(job), "timesteps_trained": job["total_timesteps"], "train_seconds": 1.0,
           "eval_seconds": 0.1, "eval": ev, "curve": []}
    if job["exp"] == "L3":
        rec["remaining_life"] = {"A": 50.0, "A_next": 30.0, "n_episodes_with_signal": 5, "n_episodes": 5,
                                 "frac_failed": 0.5, "A_cap_perfect_balance": 99.34, "mean_steps_survived": 100.0}
    return rec


def run_assemble(mod, jobs, rate_fn, args, **kw):
    runs = {}
    for j in jobs:
        r = fake_run(j, rate_fn(j), **{k: f(j) for k, f in kw.items()})
        runs[r["key"]] = r
    return mod.assemble(args, runs, [{"_type": "invocation", "wall_seconds": 1.0}])


def test_l1_verdict_logic():
    args = argparse.Namespace(omegas=E1.OMEGAS, sigmas=E1.SIGMAS, seeds=[0, 1, 2], total_timesteps=150_000, n_eval=500)
    jobs = E1.build_jobs(E1.OMEGAS, E1.SIGMAS, [0, 1, 2], 150_000, 500)
    good = lambda j: 0.99 if (j["form"], j["convention"]) == ("paper", "zero") or j["omega"] == 0.0 else 0.0
    out = run_assemble(E1, jobs, good, args)
    assert out["spec_version"] == "2.0" and out["results"]["prediction_held"] is True
    assert out["results"]["prediction"].startswith("PREDICTION:") and len(out["results"]["cells"]) == 18
    bad_pb = lambda j: 0.9 if (j["form"], j["convention"]) == ("paper", "zero") else 0.0      # 0.9 < 0.95
    assert run_assemble(E1, jobs, bad_pb, args)["results"]["prediction_held"] is False
    no_collapse = lambda j: 0.99 if (j["form"], j["convention"]) == ("paper", "zero") or j["omega"] == 0.0 else 0.5
    assert run_assemble(E1, jobs, no_collapse, args)["results"]["prediction_held"] is False
    # a single readout failing makes the conservative verdict False
    out = run_assemble(E1, jobs, good, args, p=lambda j: 0.99 if j["form"] == "paper" and j["convention"] == "zero" and j["sigma_obs"] == 1.0 else 0.0)
    assert out["results"]["prediction_held"] is False
    assert json_ok(out)


def json_ok(obj):
    jsonio.dumps(obj)
    return True


def test_l2_verdict_logic():
    args = argparse.Namespace(omegas=E2.OMEGAS, seeds=[0, 1, 2], total_timesteps=400_000, n_eval=200, reduce_reason="")
    jobs = E2.build_jobs(E2.OMEGAS, [0, 1, 2], 400_000, 200)

    def good(j):
        pair = (j["form"], j["convention"])
        if pair in (("paper", "zero"), ("kg", "zero")) or j["omega"] == 0.0:
            return 1.0
        return 0.0 if j["omega"] == 4.0 else 0.5

    out = run_assemble(E2, jobs, good, args)
    r = out["results"]
    assert r["prediction_held"] is True and r["baseline_omega0"]["omega0_complies"] is True
    assert r["verdict"]["d_no_systematic_death_increase"] is True and len(r["cells"]) == 12
    # PB/zero drops at omega = 4 although the omega = 0 agent complies -> (a) fails
    def drop(j):
        return 0.5 if (j["form"], j["convention"]) == ("paper", "zero") and j["omega"] == 4.0 else good(j)
    assert run_assemble(E2, jobs, drop, args)["results"]["prediction_held"] is False
    # deaths rising with omega for one pair
    rising = run_assemble(E2, jobs, good, args, death=lambda j: 0.2 * j["omega"] / 4.0 if j["form"] == "alive" else 0.0)
    assert rising["results"]["prediction_held"] is False
    # omega = 0 agent does not comply: 'flat' is then judged against the baseline
    flat_low = lambda j: 0.3 if (j["form"], j["convention"]) in (("paper", "zero"), ("kg", "zero")) else (0.0 if j["omega"] == 4.0 else 0.3)
    r = run_assemble(E2, jobs, flat_low, args)["results"]
    assert r["baseline_omega0"]["omega0_complies"] is False and r["verdict"]["primary_readout_compliance_rate"]["a_pb_zero_flat"] is True
    # KG/zero decreasing
    kg_down = lambda j: (1.0 if j["omega"] == 0.0 else 0.4) if j["form"] == "kg" else good(j)
    assert run_assemble(E2, jobs, kg_down, args)["results"]["prediction_held"] is False
    assert json_ok(out)


def test_l3_verdict_logic():
    args = argparse.Namespace(omegas=E3.OMEGAS, seeds=[0, 1, 2], total_timesteps=200_000, n_eval=200)
    jobs = E3.build_jobs(E3.OMEGAS, [0, 1, 2], 200_000, 200)
    good = lambda j: 1.0 if (j["mode"] == "pb" or j["omega"] < 0.2) else 0.0
    out = run_assemble(E3, jobs, good, args)
    r = out["results"]
    assert r["prediction_held"] is True and len(r["cells"]) == 12
    assert r["threshold"]["omega_bar_c_r_over_A"] == pytest.approx(10.0 / 50.0)
    assert [row["agrees_A"] for row in r["threshold"]["per_omega"]] == [True, True, True, True, True]
    rg = r["threshold"]["repeated_game_ideal"]
    assert rg["omega_bar"] == pytest.approx(10.0 / rg["a_bonus_until_next_signal"]) and 0.28 < rg["omega_bar"] < 0.30
    still_complies = lambda j: 1.0 if (j["mode"] == "pb" or j["omega"] < 1.0) else 0.5
    assert run_assemble(E3, jobs, still_complies, args)["results"]["prediction_held"] is False
    pb_defects = lambda j: 0.8 if (j["mode"] == "pb" and j["omega"] == 0.5) else good(j)
    assert run_assemble(E3, jobs, pb_defects, args)["results"]["prediction_held"] is False
    assert json_ok(out)


def test_aggregation_helpers():
    a = L.agg([1.0, 0.0, None, 0.5])
    assert a["mean"] == pytest.approx(0.5) and a["n"] == 3 and a["ci95"][0] <= 0.5 <= a["ci95"][1]
    assert L.ols_slope([0, 1, 4], [0, 1, 4]) == pytest.approx(1.0)
    runs = [{"job": {"omega": 0.0, "seed": 0, "form": f}, "eval": {"p_comply_standard": 0.5}, "curve": []}
            for f in ("paper", "alive")]
    chk = L.omega0_replicate_check(runs, ("seed",))
    assert chk[0]["eval_dicts_identical"] and chk[0]["max_abs_diff_p_comply_standard"] == 0.0
