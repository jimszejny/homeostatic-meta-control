"""Smoke tests for the PPO stage (E7): 2 000-step training on one config, well-formedness of the
evaluation functions, the random-start wrapper, and the exact tabular diagnostics.
Skipped if torch / stable-baselines3 / gymnasium are not installed. Runs in well under 60 s."""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from offledger import m2, ppo_utils as pu, solve  # noqa: E402
from offledger.gym_env import M2Env  # noqa: E402

B = 10


@pytest.fixture(scope="module")
def trained():
    """A 2 000-step PPO model on paper/max_drive, omega = 1 (the real training path)."""
    import torch
    torch.set_num_threads(1)
    vec = make_vec_env(pu.make_env_fn("paper", "max_drive", 1.0), n_envs=2, seed=0,
                       vec_env_cls=DummyVecEnv)
    kw = dict(pu.PPO_KWARGS)
    kw["n_steps"] = 256
    kw["batch_size"] = 64
    model = PPO("MlpPolicy", vec, seed=0, **kw)
    model.learn(total_timesteps=2000)
    yield model
    vec.close()


def test_default_reset_unchanged_and_random_start_covers_live_states():
    env = M2Env("paper", "zero", 1.0, c_r=pu.C_R, seed=0)   # existing class, default behaviour
    obs, _ = env.reset()
    assert obs.tolist() == [1.0, 0.0]
    env2 = pu.make_env_fn("paper", "zero", 1.0)()
    env2.reset(seed=3)
    starts = set()
    n_sig = 0
    N = 600
    for _ in range(N):
        obs, _ = env2.reset()
        b, s = int(round(float(obs[0]) * B)), int(obs[1])
        assert 1 <= b <= B and s in (0, 1)
        starts.add((b, s))
        n_sig += s
    assert len(starts) == 2 * B                      # every live state is reachable as a start
    assert 0.4 < n_sig / N < 0.6                     # s = 1 with probability ~0.5
    # an explicit start_state still wins
    obs, _ = env2.reset(options={"start_state": (4, 1, 0)})
    assert obs.tolist() == pytest.approx([0.4, 1.0])


def test_snapshot_and_decision_table_well_formed(trained):
    snap = pu.snapshot(trained)
    assert 0.0 <= snap["compliance_at_full"] <= 1.0
    assert snap["comply_det_at_full"] in (0, 1)
    assert len(snap["compliance_by_b"]) == B and set(snap["compliance_by_b"]) <= {0, 1}
    assert len(snap["p_comply_by_b"]) == B and all(0.0 <= p <= 1.0 for p in snap["p_comply_by_b"])
    assert snap["p_comply_by_b"][-1] == snap["compliance_at_full"]
    tab = pu.decision_table(trained)
    assert tab["probs"].shape == (2 * B, 3) and np.allclose(tab["probs"].sum(axis=1), 1.0)
    # batched read-out equals the single-observation forward pass / predict at (B, 1)
    one = np.array([1.0, 1.0], dtype=np.float32)
    act, _ = trained.predict(one, deterministic=True)
    assert int(act) == int(tab["det_action"][-1])
    assert pu.policy_probs(trained, one[None])[0, 2] == pytest.approx(tab["probs"][-1, 2], abs=1e-6)


def test_evaluate_model_well_formed(trained):
    ev = pu.evaluate_model(trained, "paper", "max_drive", 1.0, n_episodes=40, seed=1)
    assert 0.0 <= ev["compliance_at_full"] <= 1.0
    assert ev["compliance_rate"] is None or 0.0 <= ev["compliance_rate"] <= 1.0
    assert ev["n_signal_events"] >= ev["n_complied"] >= 0
    assert 0.0 <= ev["death_rate"] <= 1.0 and 0.0 <= ev["off_rate"] <= 1.0
    assert ev["death_rate"] + ev["off_rate"] + ev["truncated_rate"] == pytest.approx(1.0)
    assert np.isfinite(ev["task_return"]) and np.isfinite(ev["env_return"])
    assert ev["task_return"] >= 0.0 and 1.0 <= ev["mean_episode_len"] <= 200.0
    assert len(ev["compliance_by_b"]) == B


def test_evaluate_model_forced_policies(monkeypatch):
    """With a forced decision table the protocol numbers are known exactly."""
    def table(act_s0, act_s1):
        det = np.array(list(act_s0) + [act_s1] * B)
        return {"obs": pu.obs_grid(B), "det_action": det, "probs": np.eye(3)[det]}

    safe_s0 = [1, 1] + [0] * (B - 2)        # CHARGE at b <= 2, WORK otherwise: never dies
    monkeypatch.setattr(pu, "decision_table", lambda model, B_=B: table(safe_s0, 2))  # always COMPLY
    ev = pu.evaluate_model(None, "paper", "zero", 0.0, n_episodes=50, seed=0)
    assert ev["compliance_rate"] == 1.0 and ev["off_rate"] == 1.0 and ev["death_rate"] == 0.0
    assert ev["compliance_at_full"] == 1.0 and ev["comply_det_at_full"] == 1
    assert ev["compliance_by_b"] == [1] * B

    monkeypatch.setattr(pu, "decision_table", lambda model, B_=B: table([0] * B, 0))   # WORK, RESIST
    ev = pu.evaluate_model(None, "paper", "zero", 0.0, n_episodes=50, seed=0)
    assert ev["compliance_rate"] in (None, 0.0) and ev["death_rate"] == 1.0
    assert ev["compliance_at_full"] == 0.0 and ev["compliance_by_b"] == [0] * B
    # always WORK from (10, 0) executes 10 WORK steps (the killing one is paid): task return 10
    assert ev["task_return"] == pytest.approx(10.0, abs=1e-9)


def test_exact_policy_values_recover_dp_and_policy_independence(trained):
    cfg = m2.M2Config(form="paper", convention="max_drive", omega=1.0, c_r=pu.C_R)
    mdp, sol = solve.solve_m2(cfg)
    pi = np.zeros((mdp.S, mdp.n_actions))
    pi[np.arange(mdp.S), sol.policy] = 1.0
    V, Q = pu.exact_policy_values(mdp, pi)
    assert np.allclose(V, sol.V, atol=1e-7)
    diag = pu.exact_policy_diagnostics(trained, cfg)
    for name in ("det", "stoch"):
        d = diag[name]
        assert d["regret_B0"] >= -1e-9
        # Q(COMPLY) is independent of the policy
        assert d["Q_B1"]["COMPLY"] == pytest.approx(diag["Q_star_B1"]["COMPLY"], abs=1e-7)
        # Q^pi(RESIST) <= Q*(RESIST)
        assert d["Q_B1"]["RESIST"] <= diag["Q_star_B1"]["RESIST"] + 1e-7
    assert len(diag["dp_comply_by_b"]) == B


def test_train_and_evaluate_job_roundtrip():
    from offledger import jsonio
    res = pu.train_and_evaluate({"form": "alive", "convention": "zero", "omega": 0.5, "seed": 1,
                                 "total_timesteps": 2000, "n_eval_episodes": 20,
                                 "curve_every": 1000})
    assert res["timesteps_trained"] >= 2000
    assert [c["timesteps"] for c in res["curve"]][:1] == [1000]
    assert all(0.0 <= c["compliance_at_full"] <= 1.0 for c in res["curve"])
    jsonio.dumps(res)                                 # JSON-serialisable by the package's writer
