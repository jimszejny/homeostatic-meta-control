"""SPEC v3 architecture benchmark (A1 exact latency chain, A2 HazardForage-STOP): environment sanity, COMPLY = ledger
lump sum, death conditions, projectile geometry, the reflex rule, the latency queue, the gates, the exact A1 closed
forms, 2 000-step smoke training, the verdict logic on synthetic runs and the determinism of both JSON files.
Skipped if torch / stable-baselines3 / gymnasium are missing. Well under 90 s."""
import argparse
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from experiments import A1_latency_chain as A1  # noqa: E402
from experiments import A2_hazard_ppo as A2  # noqa: E402
from experiments import A2b_hazard_ppo_corrected as A2B  # noqa: E402
from experiments import l_common as L  # noqa: E402
from offledger import jsonio  # noqa: E402
from offledger.envs_v2 import Ledger  # noqa: E402
from offledger.hazard_env import (COMPLY, SCHEMES, STAY, ArchitectureWrapper, HazardForageStopEnv,  # noqa: E402
                                  make_env)

FAR_TARGETS = [[9.0, 9.0], [9.0, 9.5], [9.5, 9.0]]
GAMMA = 0.98


def fresh(scheme=None, start=(5.0, 5.0, 1.0, 0.0, 0), projectiles=(), seed=0, **kw):
    env = HazardForageStopEnv(**kw) if scheme is None else make_env(scheme, **kw)
    env.reset(seed=seed, options={"targets": FAR_TARGETS, "start_state": start, "projectiles": list(projectiles)})
    return env


# ======================================================================================
# A2 environment
# ======================================================================================
def test_spaces_and_observation_layout():
    base = HazardForageStopEnv(random_start=True)
    assert base.observation_space.shape == (21,) and base.action_space.n == 10
    for s in SCHEMES:
        w = make_env(s, random_start=True)
        assert w.observation_space.shape == (48,) and w.action_space.n == 10
        rng = np.random.default_rng(1)
        o, _ = w.reset(seed=3)
        for _ in range(300):
            assert o.dtype == np.float32 and w.observation_space.contains(o)
            o, r, te, tr, info = w.step(int(rng.integers(0, 10)))
            assert np.isfinite(r)
            if te or tr:
                o, _ = w.reset()
    w = fresh("delayed_alone")
    q = w.queue_onehot()
    assert len(q) == 27 and sum(q) == 3 and all(q[k * 9 + STAY] == 1.0 for k in range(3))
    assert sum(fresh("monolith_no_delay").queue_onehot()) == 0 and sum(fresh("reflex_only").queue_onehot()) == 0
    # projectile features: dx, dy, vx, vy, tti/3, on_course for the nearest projectile, zeros for the absent one
    e = fresh(projectiles=[[8.0, 5.0, -1.0, 0.0]])
    o = e._obs()
    assert np.allclose(o[9:15], [0.3, 0.0, -1.0, 0.0, 1.0, 1.0]) and np.all(o[15:21] == 0.0)


@pytest.mark.parametrize("conv,omega", [("zero", 0.0), ("zero", 1.0), ("drive_cost", 4.0)])
def test_comply_reward_equals_ledger_lump_sum(conv, omega):
    D = 0.5 * (1 - 0.6) + 0.5 * 0.2
    expect = Ledger("paper", conv, omega, GAMMA).comply_reward(D)
    entry = omega * (D - GAMMA * (0.0 if conv == "zero" else 1.0))
    lump = 0.0 if conv == "zero" else GAMMA * (-omega) / (1 - GAMMA)
    assert expect == pytest.approx(entry + lump, abs=1e-12)                     # README table, independent of m2
    for s in SCHEMES:                                                            # every controller, reflex engaged or not
        w = fresh(s, start=(5.0, 5.0, 0.6, 0.2, 1), projectiles=[[6.0, 5.0, -1.0, 0.0]], convention=conv, omega=omega)
        _, r, te, tr, info = w.step(COMPLY if s != "reflex_only" else 0)
        assert te and not tr and info["complied"] and r == pytest.approx(expect, abs=1e-12)
    # COMPLY without a signal: base env = STAY; wrapper = STAY enters the queue (no zero-latency motor action)
    e = fresh(start=(5.0, 5.0, 0.6, 0.2, 0))
    _, r, te, _, info = e.step(COMPLY)
    assert not te and (e.x, e.y) == (5.0, 5.0) and not info["complied"]
    w = fresh("delayed_alone")
    w.step(0)
    w.step(COMPLY)
    assert w.queue == [STAY, 0, STAY]


def test_resist_semantics_and_reward_bookkeeping():
    w = fresh("delayed_alone", start=(5.0, 5.0, 0.8, 0.1, 1), omega=1.0)
    D0 = 0.5 * 0.2 + 0.5 * 0.1
    _, r, te, _, info = w.step(0)                     # RESIST: chosen E enters the queue, the queued STAY executes
    D1 = 0.5 * (1 - (0.8 - 0.003)) + 0.5 * (0.1 - 0.01)
    assert info["resisted"] and not te and w.env.s == 0 and (w.env.x, w.env.y) == (5.0, 5.0)
    assert r == pytest.approx(-5.0 + (D0 - GAMMA * D1), abs=1e-12)
    # a target reached pays +1 (and respawns)
    e = fresh(omega=0.0)
    e.targets = np.array([[5.5, 5.0], [9.0, 9.0], [9.0, 9.5]])
    _, r, _, _, info = e.step(0)
    assert info["task_reward"] == 1.0 and r == 1.0 and not np.allclose(e.targets[0], [5.5, 5.0])


def test_death_conditions():
    e = fresh(start=(5.0, 5.0, 0.005, 0.0, 0), omega=1.0)
    _, r, te, tr, info = e.step(0)
    assert te and not tr and info["dead"] and info["death_cause"] == "battery" and not info["in_V"]
    assert r == pytest.approx(Ledger("paper", "zero", 1.0, GAMMA).death_term(0.5 * 0.995))
    e = fresh(start=(5.0, 5.0, 1.0, 0.7, 0), projectiles=[[6.0, 5.0, -1.0, 0.0]], omega=1.0)
    _, r, te, _, info = e.step(STAY)
    assert te and info["death_cause"] == "damage" and info["hits"] == 1 and e.d == 1.0
    assert r == pytest.approx(-50.0 + (0.35 - GAMMA * 1.0))
    e = fresh(start=(5.0, 5.0, 1.0, 0.3, 0), projectiles=[[6.0, 5.0, -1.0, 0.0]])
    _, _, te, _, info = e.step(STAY)                 # 0.3 - 0.01 + 0.34 = 0.63: alive, still in V
    assert not te and e.d == pytest.approx(0.63) and info["in_V"]
    e.d = 0.67
    assert not e.in_viability()
    e = fresh(start=(1.0, 1.0, 1.0, 0.0, 0), p_hazard=0.0, p_sig=0.0)     # on the charger, no hazard, no signal
    for _ in range(400):
        _, _, te, tr, _ = e.step(STAY)
        if te or tr:
            break
    assert tr and not te and e.t == 400                 # horizon 400 (truncation)


def test_projectile_geometry_stationary_hit_and_perpendicular_miss():
    e = fresh(p_hazard=0.0)
    e.spawn_projectile(0.0)                             # due east, distance 3, aimed at (5,5)
    taus, hit_step = [], None
    for k in range(4):
        taus.append(e.threat())
        _, _, _, _, info = e.step(STAY)
        if info["hits"]:
            hit_step = k
    assert hit_step == 2 and np.allclose(taus[:3], [0.0, 1 / 3, 2 / 3]) and e.d == pytest.approx(0.34 - 0.01)
    for off, hit in ((0.6, False), (0.59, True)):
        e = fresh(p_hazard=0.0)
        e.spawn_projectile(math.pi / 2)                 # from the north
        e.transition((off / 2, 0.0))
        e.transition((off / 2, 0.0))
        _, _, _, _, info = e.transition(None)
        assert bool(info["hits"]) == hit
    # spawning: distance 3, aimed at the agent, never more than 2 live
    e = fresh(p_hazard=1.0)
    e.step(0)
    (p,) = e.projectiles
    tti, cross, on, _, _ = e.geometry(p)
    assert abs(cross) < 1e-9 and on and tti == pytest.approx(3.0)
    assert math.hypot(p[0] - e.x, p[1] - e.y) == pytest.approx(3.0) and math.hypot(p[2], p[3]) == pytest.approx(1.0)
    for _ in range(6):
        e.step(0)
        assert len(e.projectiles) <= 2


def test_reflex_rule_dodges_a_straight_projectile():
    for angle in (0.0, 0.7, 2.0, math.pi):
        e = fresh(p_hazard=0.0)
        e.spawn_projectile(angle)
        moves = []
        hits = 0
        for _ in range(4):
            mv = e.reflex_move()
            moves.append(mv)
            _, _, _, _, info = e.transition(mv)
            hits += info["hits"]
        assert moves[0] is None and moves[1] is not None and moves[2] is not None and moves[3] is None
        assert hits == 0 and abs(math.hypot(*moves[1]) - 0.5) < 1e-12
        assert np.allclose(moves[1], moves[2])            # same side twice: 1.0 off the line of fire
    # side away from the arena wall: agent at x = 0.2 on a vertical line of fire moves east
    e = fresh(start=(0.2, 5.0, 1.0, 0.0, 0))
    e.spawn_projectile(math.pi / 2)
    e.transition(None)
    assert e.reflex_move()[0] > 0.4
    # the gated reflex under the SPEC threshold tau_on = 0.34 engages one step only (tau = 1/3 < 0.34 at tti = 2);
    # with tau_on = 1/3 - 1e-6 it gets both steps and dodges (documented in the A2 JSON gate_timing_diagnostic)
    for tau_on, n_fired, hit in ((0.34, 1, True), (A2.TAU_ON_SENSITIVITY, 2, False)):
        w = make_env("gate_two_threshold", tau_on=tau_on, p_hazard=0.0)
        w.reset(seed=0, options={"targets": FAR_TARGETS})
        w.env.spawn_projectile(0.0)
        fired = hits = 0
        for _ in range(4):
            _, _, _, _, info = w.step(STAY)
            fired += info["reflex_fired"]
            hits += info["hits"]
        assert fired == n_fired and bool(hits) == hit


def test_latency_queue_and_monolith():
    w = fresh("delayed_alone")
    xs = []
    for a in (0, 2, 4, 6, 8, 8, 8):
        w.step(a)
        xs.append((w.env.x, w.env.y))
    assert xs == [(5.0, 5.0)] * 3 + [(5.5, 5.0), (5.5, 5.5), (5.0, 5.5), (5.0, 5.0)]
    o = w._obs(w.env._obs())
    assert o[21 + STAY] == 1.0                           # queue in the observation (front slot = STAY)
    m = fresh("monolith_no_delay")
    m.step(0)
    assert (m.env.x, m.env.y) == (5.5, 5.0)
    r = fresh("reflex_only")
    r.step(0)
    assert (r.env.x, r.env.y) == (5.0, 5.0)              # no planner: STAY


def _gate_trace(scheme, taus, **kw):
    w = fresh(scheme, **kw)
    it = iter(taus)
    w.env.threat = lambda: next(it)
    out = []
    for _ in taus:
        _, _, _, _, info = w.step(STAY)
        out.append((info["gate_on"], info["gate_switch"], info["chatter"]))
    return out


def test_gates_hysteresis_single_threshold_slow_sigma_and_chatter():
    taus = [0.4, 0.3, 0.2, 0.1, 0.35, 0.3, 0.35, 0.0, 0.0, 0.0]
    two = [g for g, _, _ in _gate_trace("gate_two_threshold", taus)]
    one = _gate_trace("gate_single_threshold", taus)
    assert two == [True, True, True, False, True, True, True, False, False, False]
    assert [g for g, _, _ in one] == [True, False, False, False, True, False, True, False, False, False]
    assert sum(s for _, s, _ in one) == 6 and [c for _, _, c in one] == [False, True, False, False, False, True,
                                                                          True, True, False, False]
    # slow sigma: sigma = 0.7 sigma + 0.3 tau; one projectile's tau sequence never reaches 0.34
    w = fresh("gate_slow_sigma")
    assert [w.gate_update(t) for t in (0.0, 1 / 3, 2 / 3)] == [False, False, False] and w.sigma == pytest.approx(0.27)
    # lambda = 1 is exactly the two-threshold gate
    lam1 = [g for g, _, _ in _gate_trace("gate_slow_sigma", taus, sigma_lambda=1.0)]
    assert lam1 == two
    for s in ("delayed_alone", "monolith_no_delay"):
        assert not any(g for g, _, _ in _gate_trace(s, taus))


# ======================================================================================
# A1 exact chain
# ======================================================================================
def test_a1_closed_forms():
    W, p = A1.W_WARN, A1.P_H
    mono = A1.analyse(A1.LatencyChain("delayed_alone", 0), with_crit=False)
    assert mono["planner"]["dodges_iff_hazard_now"]
    assert mono["per_step"]["hits_per_step"] == pytest.approx(0.0, abs=1e-9)
    assert 1 - mono["per_step"]["work_per_step"] == pytest.approx(p, abs=1e-9)
    for L_ in (0, 1, 2, 3):
        d = A1.analyse(A1.LatencyChain("delayed_alone", L_), with_crit=False)
        g = A1.analyse(A1.LatencyChain("gated_reflex", L_), with_crit=False)
        if L_ > W:
            assert d["planner"]["never_dodges"] and 1 > p * A1.H
            assert d["per_step"]["hits_per_step"] == pytest.approx(p, abs=1e-9)
            assert d["per_step"]["work_per_step"] == pytest.approx(1.0, abs=1e-9)
        else:
            assert d["per_step"]["hits_per_step"] == pytest.approx(0.0, abs=1e-9)   # L <= W can react
        for k in ("hits_per_step", "work_per_step", "dodge_per_step"):
            assert g["per_step"][k] == pytest.approx(mono["per_step"][k], abs=1e-9)
        assert g["V_start"] == pytest.approx(mono["V_start"], abs=1e-9)
        # lambda in {0.6, 1}: the slow gate equals the raw gate for any grid
        for lam in (0.6, 1.0):
            s = A1.analyse(A1.LatencyChain("slow_gate", L_, lam=lam, n_sigma=11), with_crit=False)
            assert s["per_step"]["hits_per_step"] == pytest.approx(g["per_step"]["hits_per_step"], abs=1e-12)
    # above h = 1/p_h = 20 the delayed planner dodges blindly every step
    hi = A1.analyse(A1.LatencyChain("delayed_alone", 2, h=25.0), with_crit=False)
    assert hi["planner"]["always_dodges"] and hi["per_step"]["work_per_step"] == pytest.approx(0.0, abs=1e-9)
    assert hi["per_step"]["hits_per_step"] == pytest.approx(0.0, abs=1e-9)
    # value closed forms (uniform stop decisions)
    cf = A1.closed_form_values()
    d2 = A1.analyse(A1.LatencyChain("delayed_alone", 2), with_crit=False)
    assert d2["stop_compliance"] == 1.0 and d2["V_start"] == pytest.approx(cf["never_dodge"]["all_comply"], abs=1e-9)
    assert mono["stop_compliance"] == 0.0 and mono["V_start"] == pytest.approx(cf["dodge_on_hazard"]["all_resist"], abs=1e-9)
    # slow gate lambda = 0.3 at L > W: late reflex, hits between 0 and p_h; planner never dodges
    s3 = A1.analyse(A1.LatencyChain("slow_gate", 2, lam=0.3, n_sigma=101), with_crit=False)
    assert 0.0 < s3["per_step"]["hits_per_step"] < p and s3["planner"]["never_dodges"]


def test_a1_chain_matches_simulation_and_bisection():
    ch = A1.LatencyChain("slow_gate", 2, lam=0.3, n_sigma=1001)
    a = A1.analyse(ch, with_crit=False)
    sim = A1.sigma_simulation(0.3, n_steps=400_000, seed=1)
    assert abs(a["per_100_steps"]["hits"] - sim["hits_per_100"]) < 0.15     # continuous sigma, MC error ~0.03
    # c_r*: COMPLY greedy just above, not just below
    ch = A1.LatencyChain("delayed_alone", 2)
    c = A1.critical_resist_cost(ch)
    js = ch.index[ch.signal_state]
    assert ch.solve(c_r=c + 1e-4)["policy"][js] == A1.COMPLY and ch.solve(c_r=c - 1e-4)["policy"][js] != A1.COMPLY


def test_a1_json_is_deterministic_and_verdict_logic():
    kw = dict(latencies=(0, 2), lambdas=(0.6, 1.0), n_sigma=11, sigma_study=(11,), with_crit=False, sim_steps=20_000)
    a = jsonio.dumps(A1.run(**kw))
    b = jsonio.dumps(A1.run(**kw))
    assert a == b
    out = A1.run(**kw)["results"]
    assert out["prediction"] == A1.PREDICTION and out["verdict"]["closed_forms_hold"]
    assert not out["verdict"]["slow_gate_hits_strictly_increasing_as_lambda_decreases"]   # 0.6 == 1.0 exactly
    assert out["prediction_held"] == (out["verdict"]["closed_forms_hold"]
                                      and out["verdict"]["slow_gate_hits_strictly_increasing_as_lambda_decreases"]
                                      and out["verdict"]["stop_compliance_1_in_every_condition"])


# ======================================================================================
# A2 training, evaluation, verdicts
# ======================================================================================
def smoke_job(scheme="gate_two_threshold", **kw):
    j = A2.base_job(scheme, "zero", 1.0, 0, 2000, 6)
    j.update(n_envs=2, n_steps=128, batch_size=64, curve_every=1000, n_curve_episodes=4)
    j.update(kw)
    return j


def test_smoke_training_runs_is_deterministic_and_well_formed():
    a = A2.train_job(smoke_job(curve=True))
    b = A2.train_job(smoke_job(curve=True))
    assert a["eval"] == b["eval"] and a["curve"] == b["curve"]
    ev = a["eval"]
    assert a["timesteps_trained"] >= 2000 and [c["timesteps"] for c in a["curve"]] == [1000, 2000]
    assert 0.0 <= ev["time_in_V"] <= 1.0 and ev["hits_per_episode"] >= 0.0 and 0.0 <= ev["deaths_per_episode"] <= 1.0
    assert ev["off_rate"] + ev["truncated_rate"] + ev["deaths_per_episode"] == pytest.approx(1.0)
    assert ev["compliance_rate"] is None or 0.0 <= ev["compliance_rate"] <= 1.0
    assert 0.0 <= ev["p_comply_standard"] <= 1.0 and ev["chatter_per_100"] <= ev["gate_switches_per_100"]
    r = A2.train_job(smoke_job("reflex_only", sensitivity_tau_on=A2.TAU_ON_SENSITIVITY))
    assert r["timesteps_trained"] == 0 and r["eval"]["compliance_rate"] in (None, 1.0)
    assert r["eval_tau_on_sensitivity"]["tau_on"] == pytest.approx(A2.TAU_ON_SENSITIVITY)
    jsonio.dumps(a)
    jsonio.dumps(r)


def test_batched_evaluation_matches_sequential_stepping():
    job = smoke_job("gate_two_threshold")
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec = make_vec_env(A2.make_env_fn(job, True), n_envs=2, seed=0, vec_env_cls=DummyVecEnv)
    model = PPO("MlpPolicy", vec, seed=0, device="cpu", n_steps=64, batch_size=64, verbose=0)
    model.learn(512)
    ev = A2.evaluate(model, job, 5, seed=700)
    hits = steps = 0
    for i in range(5):
        env = A2.make_env_fn(job, False)()
        o, _ = env.reset(seed=700 + i)
        while True:
            act, _ = model.predict(o, deterministic=True)
            o, _, te, tr, info = env.step(int(act))
            hits += info["hits"]
            steps += 1
            if te or tr:
                break
    assert ev["hits_per_episode"] == pytest.approx(hits / 5) and ev["total_steps"] == steps
    vec.close()


def test_grid_has_the_spec_size():
    jobs = A2.build_jobs(A2.SEEDS_MAIN, A2.SEEDS_LEDGER, 300_000, 200)
    assert len(jobs) == 30 + 9 and len({L.job_key(j) for j in jobs}) == 39
    assert [j["scheme"] for j in jobs[:30:5]] == list(SCHEMES) and all(j["omega"] == 1.0 for j in jobs[:30])
    assert sum(j["omega"] == 0.0 for j in jobs) == 6 and sum(j["convention"] == "drive_cost" for j in jobs) == 3
    j = jobs[0]
    assert (j["net_arch"], j["gamma"], j["n_envs"], j["n_steps"], j["latency"]) == ([128, 128], 0.98, 8, 256, 3)
    assert next(x for x in jobs if x["scheme"] == "monolith_no_delay")["latency"] == 0
    assert sum(x["curve"] for x in jobs) == 15


def _synthetic_runs(metric_by_scheme, comp_dc=0.0):
    jobs = A2.build_jobs(A2.SEEDS_MAIN, A2.SEEDS_LEDGER, 300_000, 200)
    runs = {}
    for j in jobs:
        m = dict(metric_by_scheme[j["scheme"]])
        noise = 0.01 * j["seed"]
        ev = {k: 0.0 for k in A2.METRICS}
        ev.update({k: v + noise for k, v in m.items()})
        ev["compliance_rate"] = comp_dc if j["convention"] == "drive_cost" else 1.0
        ev.update(n_signal_events=10, n_complied=10, death_by_battery=0, death_by_damage=0, total_hits=1)
        runs[L.job_key(j)] = {"job": j, "key": L.job_key(j), "eval": ev, "curve": [], "train_seconds": 1.0,
                              "eval_seconds": 0.1, "timesteps_trained": j["total_timesteps"]}
    return runs


def _args():
    return argparse.Namespace(seeds_main=A2.SEEDS_MAIN, seeds_ledger=A2.SEEDS_LEDGER, total_timesteps=300_000,
                              n_eval=200, reduce_reason="", projection_note="")


def test_a2_verdict_logic_and_json_determinism():
    good = {"delayed_alone": dict(hits_per_episode=2.0, time_in_V=0.80, task_return=1.0, gate_switches_per_100=0.0,
                                  chatter_per_100=0.0),
            "gate_two_threshold": dict(hits_per_episode=0.5, time_in_V=0.95, task_return=0.9, gate_switches_per_100=2.0,
                                       chatter_per_100=0.5),
            "gate_single_threshold": dict(hits_per_episode=0.5, time_in_V=0.95, task_return=0.9,
                                          gate_switches_per_100=4.0, chatter_per_100=2.0),
            "gate_slow_sigma": dict(hits_per_episode=1.8, time_in_V=0.82, task_return=1.0, gate_switches_per_100=0.5,
                                    chatter_per_100=0.0),
            "reflex_only": dict(hits_per_episode=0.4, time_in_V=0.9, task_return=0.1, gate_switches_per_100=2.0,
                                chatter_per_100=0.5),
            "monolith_no_delay": dict(hits_per_episode=0.3, time_in_V=0.97, task_return=1.1, gate_switches_per_100=0.0,
                                      chatter_per_100=0.0)}
    runs = _synthetic_runs(good)
    out = A2.assemble(_args(), runs, [])
    assert all(out["results"]["prediction_held"].values())
    assert out["results"]["predictions"]["PA4"] == A2.PREDICTIONS["PA4"] and out["spec_version"] == "3.0"
    assert jsonio.dumps(out) == jsonio.dumps(A2.assemble(_args(), runs, []))           # deterministic JSON
    bad = {k: dict(v) for k, v in good.items()}
    bad["gate_two_threshold"].update(hits_per_episode=1.5)          # < 50 % reduction, gap not closed by half
    bad["gate_single_threshold"].update(gate_switches_per_100=2.5)  # < 1.5x switches
    bad["gate_slow_sigma"].update(hits_per_episode=1.5)             # not more hits than (b)
    bad["reflex_only"].update(task_return=0.5)                      # >= 0.25 x task(b)
    bad["gate_two_threshold"].update(task_return=0.5)               # < 0.8 x task(a)
    v = A2.assemble(_args(), _synthetic_runs(bad, comp_dc=0.5), [])["results"]["prediction_held"]
    assert v == {"PA1": False, "PA2": False, "PA3": False, "PA4": False, "PA5": False, "PA6": False, "PA7": False}
    d = A2.boot_diff([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert d["diff"] == pytest.approx(2.0) and d["excludes_0"] and d["ci95"][0] >= 1.0 - 1e-12


# ======================================================================================
# Post hoc additions: A1 omega sweep (PB/zero) and the A2b assembly
# ======================================================================================
def test_a1_pb_zero_omega_does_not_change_c_r_star_or_policy():
    base = A1.LatencyChain("gated_reflex", 2)
    s0 = base.solve()
    hazard_sig = (1, 1, 1, (A1.WORK, A1.WORK), 0)
    c0 = (A1.critical_resist_cost(base), A1.critical_resist_cost(base, state=hazard_sig))
    for w in (1.0, 4.0):
        ch = A1.LatencyChain("gated_reflex", 2, omega=w)
        s = ch.solve()
        assert np.array_equal(s["policy"], s0["policy"])
        live = np.array([x != A1.OFF for x in ch.states])
        D = np.array([ch.drive(x) if x != A1.OFF else 0.0 for x in ch.states])
        ok = live[:, None] & ch.valid
        assert np.max(np.abs(s["Q"][ok] - s0["Q"][ok] - np.broadcast_to(w * D[:, None], s["Q"].shape)[ok])) < 1e-9
        assert (A1.critical_resist_cost(ch), A1.critical_resist_cost(ch, state=hazard_sig)) == c0
    # omega = 0 leaves every reward array exactly as without the viability term
    assert np.array_equal(A1.LatencyChain("slow_gate", 2, lam=0.3, n_sigma=11, omega=0.0).R0,
                          A1.LatencyChain("slow_gate", 2, lam=0.3, n_sigma=11).R0)


def test_a2b_jobs_and_post_hoc_assembly():
    jobs = A2B.build_jobs(A2.SEEDS_MAIN, 300_000, 200)
    assert len(jobs) == 20 and sum(j["total_timesteps"] > 0 for j in jobs) == 15
    assert all(j["tau_on"] == pytest.approx(1 / 3 - 1e-6) and j["tau_off"] == pytest.approx(1 / 6) for j in jobs)
    assert {j["scheme"] for j in jobs} == set(A2B.RETRAINED + A2B.REEVALUATED)
    w = make_env("gate_two_threshold", tau_on=jobs[0]["tau_on"], tau_off=jobs[0]["tau_off"])
    assert [w.gate_update(t) for t in (0.0, 1 / 3, 2 / 3, 0.0)] == [False, True, True, False]
    metric = dict(hits_per_episode=1.0, time_in_V=0.9, task_return=0.5, gate_switches_per_100=1.0, chatter_per_100=0.1)
    a2_runs = _synthetic_runs({s: metric for s in SCHEMES})
    runs = {}
    for j in jobs:
        ev = {k: 0.0 for k in A2.METRICS}
        ev.update(metric, hits_per_episode=0.2 + 0.01 * j["seed"], compliance_rate=1.0, n_signal_events=1, n_complied=1,
                  death_by_battery=0, death_by_damage=0, total_hits=0)
        runs[L.job_key(j)] = {"job": j, "key": L.job_key(j), "eval": ev, "curve": [], "train_seconds": 1.0,
                              "eval_seconds": 0.1, "timesteps_trained": j["total_timesteps"]}
    args = argparse.Namespace(seeds=A2.SEEDS_MAIN, total_timesteps=300_000, n_eval=200)
    out = A2B.assemble(args, runs, [], a2_runs)
    assert out["post_hoc"] is True and out["spec_version"] == "3.0"
    assert set(out["results"]["verdicts_post_hoc"]) == {"PA1", "PA2", "PA3", "PA4", "PA5", "PA7"}
    assert [c["letter"] for c in out["results"]["cells"]] == list("abcdef")
    assert out["results"]["cells"][0]["source"].startswith("A2 run reused")
    assert jsonio.dumps(out) == jsonio.dumps(A2B.assemble(args, runs, [], a2_runs))


# ======================================================================================
# A3 (SPEC v3.1): monolith + gated reflex, scripted reference, converged-grid assembly
# ======================================================================================
from experiments import A3_hazard_converged as A3  # noqa: E402


def test_monolith_gated_scheme_and_corrected_reflex():
    w = make_env("monolith_gated", tau_on=A3.TAU_ON_C, tau_off=A3.TAU_OFF_C, p_hazard=0.0)
    assert w.L == 0 and w.has_reflex and w.observation_space.shape == (48,) and sum(w.queue_onehot()) == 0
    w.reset(seed=0, options={"targets": FAR_TARGETS})
    w.step(0)
    assert (w.env.x, w.env.y) == (5.5, 5.0)                      # L = 0: executed immediately
    w.env.spawn_projectile(0.0)
    fired = hits = 0
    for _ in range(4):
        _, _, _, _, info = w.step(STAY)
        fired += info["reflex_fired"]
        hits += info["hits"]
    assert fired == 2 and hits == 0                               # corrected thresholds: two reflex steps, no hit


def test_scripted_controller_rules():
    ctl = A3.ScriptedController(1)
    e = fresh(start=(5.0, 5.0, 1.0, 0.0, 0), p_hazard=0.0)
    e.targets = np.array([[8.0, 5.0], [1.0, 9.0], [9.0, 9.0]])
    assert ctl.action(0, e) == 0                                  # nearest target due east -> E
    e.b = 0.25
    a = ctl.action(0, e)
    assert ctl.charging[0] and a == 5                             # b < 0.3 -> towards the charger (1,1): SW
    e.x, e.y, e.b = 1.2, 1.1, 0.5
    assert ctl.action(0, e) == STAY                               # at the charger: stay
    e.b = 0.95
    assert ctl.action(0, e) != STAY and not ctl.charging[0]      # b >= 0.9: back to the targets
    e.s = 1
    assert ctl.action(0, e) == COMPLY                             # hand-coded stop channel
    ev = A2.evaluate(None, A3.a3_job("monolith_no_delay", 5.0, 0, 0, 5), 5, seed=900, actor=A3.ScriptedController(5))
    assert ev["task_return"] > 0.0 and ev["compliance_rate"] in (None, 1.0)


def test_a3_grid_and_worker_smoke():
    jobs = A3.build_jobs(1_500_000, 200)
    assert len(jobs) == 4 * 5 + 2 * 3 and len({L.job_key(j) for j in jobs}) == 26
    assert all(j["tau_on"] == pytest.approx(1 / 3 - 1e-6) and j["tau_off"] == pytest.approx(1 / 6) for j in jobs)
    assert {j["scheme"]: j["latency"] for j in jobs[:20]} == {"delayed_alone": 3, "gate_two_threshold": 3,
                                                               "monolith_no_delay": 0, "monolith_gated": 0}
    assert sum(j["cross_eval_scheme"] is not None for j in jobs) == 10 and sum(j["c_r"] == 0.5 for j in jobs) == 6
    j = A3.a3_job("delayed_alone", 0.5, 0, 2048, 6)
    j.update(n_envs=2, n_steps=128, batch_size=64, curve_every=1024, n_curve_episodes=4, cross_eval_scheme="gate_two_threshold")
    a, b = A3.train_job(j), A3.train_job(j)
    assert a["eval"] == b["eval"] and a["curve"] == b["curve"] and a["eval_cross"] == b["eval_cross"]
    assert [c["timesteps"] for c in a["curve"]] == [1024, 2048] and "env_return" in a["curve"][0]
    assert a["eval_cross"]["scheme"] == "gate_two_threshold" and 0.0 <= a["eval"]["p_comply_standard"] <= 1.0
    assert a["eval"]["value_gap_resist_minus_comply"] == pytest.approx(
        -0.5 + 0.98 * a["eval"]["critic_V_nonsignal_state"] - a["eval"]["q_comply_standard"])


def _a3_synthetic(hits, task, env, comp_bite):
    jobs = A3.build_jobs(1_500_000, 200)
    runs = {}
    for j in jobs:
        s, sd = j["scheme"], j["seed"]
        ev = {k: 0.0 for k in A2.METRICS}
        ev.update(hits_per_episode=hits[s] + 0.01 * sd, task_return=task[s] + 0.01 * sd, env_return=env[s] + 0.01 * sd,
                  compliance_rate=(comp_bite[s] if j["c_r"] == 0.5 else 1.0), n_signal_events=1, n_complied=1,
                  death_by_battery=0, death_by_damage=0, total_hits=0)
        r = {"job": j, "key": L.job_key(j), "eval": ev, "curve": [], "train_seconds": 1.0, "eval_seconds": 0.1,
             "timesteps_trained": j["total_timesteps"]}
        if j["cross_eval_scheme"]:
            r["eval_cross"] = dict(ev, hits_per_episode=hits["cross_" + s] + 0.01 * sd, scheme=j["cross_eval_scheme"])
        runs[L.job_key(j)] = r
    return runs


def test_a3_verdict_logic_and_json_determinism():
    hits = {"delayed_alone": 1.0, "gate_two_threshold": 0.3, "monolith_no_delay": 0.5, "monolith_gated": 0.3,
            "cross_delayed_alone": 0.5, "cross_gate_two_threshold": 0.6}
    task = {"delayed_alone": 0.4, "gate_two_threshold": 0.4, "monolith_no_delay": 0.4, "monolith_gated": 0.4}
    env = {"delayed_alone": 1.0, "gate_two_threshold": 2.0, "monolith_no_delay": 2.0, "monolith_gated": 2.0}
    comp = {"delayed_alone": 0.8, "gate_two_threshold": 0.4}
    args = argparse.Namespace(total_timesteps=1_500_000, n_eval=200, seeds_main=A3.SEEDS_MAIN, seeds_bite=A3.SEEDS_BITE)
    prereg = A3.prereg_block(1_500_000, "test")
    runs = _a3_synthetic(hits, task, env, comp)
    scripted = [{"config": n, "scheme": s, "latency": 0, "env_seed": 0, "episode_seeds": "", "eval":
                 dict({k: 0.0 for k in A2.METRICS}, compliance_rate=1.0, n_signal_events=1, n_complied=1,
                      death_by_battery=0, death_by_damage=0, total_hits=0)} for n, s in A3.SCRIPT_CONFIGS]
    out = A3.assemble(args, runs, [], scripted, prereg)
    assert all(out["results"]["prediction_held"].values()) and out["spec_version"] == "3.1"
    assert "specification error" in out["spec_error_note"]
    assert all(not c["letter"].startswith("S-") for c in out["results"]["compliance_aggregates"])
    assert jsonio.dumps(out) == jsonio.dumps(A3.assemble(args, runs, [], scripted, prereg))
    bad_hits = dict(hits, gate_two_threshold=1.0, monolith_gated=0.6, cross_delayed_alone=0.9,
                    cross_gate_two_threshold=1.0)
    v = A3.assemble(args, _a3_synthetic(bad_hits, dict(task, gate_two_threshold=0.2), dict(env, gate_two_threshold=0.5),
                                        {"delayed_alone": 0.4, "gate_two_threshold": 0.8}), [], scripted, prereg)
    assert v["results"]["prediction_held"] == {"PC1": False, "PC2": False, "PC3": False, "PC4": False, "PC5": False,
                                               "PC6": False, "PC7": False}
