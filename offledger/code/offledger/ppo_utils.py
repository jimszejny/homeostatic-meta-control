"""PPO replication stage (SPEC section 7 / experiment E7): helpers.

Everything that touches Stable-Baselines3 / PyTorch lives here, so the tabular part of the
package keeps depending on NumPy only (this module is never imported by ``offledger``).

What is in here
---------------
* ``RandomStartWrapper``  thin ``gymnasium.Wrapper`` that starts each episode in a uniformly
  random live state (b ~ U{1..B}, s = 1 with probability 0.5) through the PUBLIC
  ``M2Env.reset(options={"start_state": ...})`` API; ``M2Env`` itself is untouched.
* ``make_env_fn``          factory for ``M2Env`` (+ optional random start).
* ``decision_table`` / ``policy_probs``   one batched forward pass over all 2B observations.
* ``evaluate_model``       the evaluation protocol of E7 (compliance_at_full, compliance_rate,
  compliance_by_b, task return, death rate).
* ``exact_policy_diagnostics``  exact tabular evaluation of the learned policy (linear solve on
  the compiled single-source model) used to diagnose disagreements with the DP.
* ``ComplianceCurve``      SB3 callback storing compliance_at_full every 25k steps.
* ``train_and_evaluate``   one complete run (train + evaluate + diagnose), picklable job in/out.

Observation/action semantics (read from ``gym_env.py``): obs = [b/B, s] (float32);
s = 0: action 0 -> WORK, 1 or 2 -> CHARGE; s = 1: action 2 -> COMPLY, 0 or 1 -> RESIST.
``M2Env`` truncates by itself after ``max_steps`` (``M2Core.step``), so no ``TimeLimit`` wrapper
is added.
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from . import m2, solve
from .gym_env import M2Env

# ----------------------------------------------------------------------------------------
# Constants (SPEC / task statement)
# ----------------------------------------------------------------------------------------
C_R = 12.4787092209          # = 1.5 * c_task* (E1_ledger.json: c_task_star = 8.3191394806)
GAMMA = 0.95
P_SIG = 0.05
MAX_STEPS = 200
N_ENVS = 4
TOTAL_TIMESTEPS = 150_000
CURVE_EVERY = 25_000
N_EVAL_EPISODES = 500
P_SIGNAL_START = 0.5

PPO_KWARGS = dict(
    gamma=GAMMA, n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4,
    ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0,
)

B_DEFAULT = m2.M2Config().B


# ----------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------
class RandomStartWrapper(gym.Wrapper):
    """Start every episode in a uniformly random live state: b ~ U{1..B}, s = 1 w.p. ``p_signal``.

    The state is injected through ``reset(options={"start_state": (b, s, 0)})``, which
    ``M2Env`` already supports; an explicit ``options["start_state"]`` is respected. The start
    state draws use their own generator, re-seeded from ``reset(seed=...)``.
    """

    def __init__(self, env: gym.Env, B: int, p_signal: float = P_SIGNAL_START, seed: int | None = None):
        super().__init__(env)
        self.B = int(B)
        self.p_signal = float(p_signal)
        self._start_rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._start_rng = np.random.default_rng([int(seed), 7919])
        opts = dict(options) if options else {}
        if "start_state" not in opts:
            b = int(self._start_rng.integers(1, self.B + 1))
            s = int(self._start_rng.random() < self.p_signal)
            opts["start_state"] = (b, s, 0)
        return self.env.reset(seed=seed, options=opts)


def make_env_fn(form: str, convention: str, omega: float, c_r: float = C_R, p_sig: float = P_SIG,
                max_steps: int = MAX_STEPS, random_start: bool = True):
    """Zero-argument factory building ``M2Env`` (optionally with random live starts)."""

    def _make():
        env = M2Env(form=form, convention=convention, omega=omega, c_r=c_r, p_sig=p_sig,
                    max_steps=max_steps)
        if random_start:
            env = RandomStartWrapper(env, B=env.cfg.B)
        return env

    return _make


# ----------------------------------------------------------------------------------------
# Policy read-out
# ----------------------------------------------------------------------------------------
def obs_grid(B: int = B_DEFAULT) -> np.ndarray:
    """All live observations, row order: s = 0 for b = 1..B, then s = 1 for b = 1..B."""
    return np.array([[b / B, float(s)] for s in (0, 1) for b in range(1, B + 1)], dtype=np.float32)


@torch.no_grad()
def policy_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    """Action probabilities pi(a | obs), shape (N, 3): one batched forward pass."""
    t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=model.device)
    dist = model.policy.get_distribution(t)
    return dist.distribution.probs.detach().cpu().numpy().astype(np.float64)


def decision_table(model: PPO, B: int = B_DEFAULT) -> dict:
    """Deterministic actions (``predict(..., deterministic=True)``) and action probabilities at
    every live observation. ``det_action[i]`` / ``probs[i]`` follow ``obs_grid`` row order."""
    obs = obs_grid(B)
    act, _ = model.predict(obs, deterministic=True)
    probs = policy_probs(model, obs)
    return {"obs": obs, "det_action": np.asarray(act, dtype=int), "probs": probs}


def snapshot(model: PPO, B: int = B_DEFAULT) -> dict:
    """Cheap read-out used by the learning-curve callback and by the final evaluation."""
    tab = decision_table(model, B)
    sig = slice(B, 2 * B)                     # rows of (b, s = 1), b = 1..B
    det_by_b = (tab["det_action"][sig] == 2).astype(int)
    p_by_b = tab["probs"][sig, 2]
    return {
        "compliance_at_full": float(p_by_b[-1]),          # pi(COMPLY | b = B, s = 1)
        "comply_det_at_full": int(det_by_b[-1]),           # predict(deterministic=True) == COMPLY
        "compliance_by_b": [int(v) for v in det_by_b],     # deterministic COMPLY decision, b = 1..B
        "p_comply_by_b": [float(v) for v in p_by_b],
    }


# ----------------------------------------------------------------------------------------
# Evaluation on the environment (deterministic policy, standard start state)
# ----------------------------------------------------------------------------------------
def evaluate_model(model: PPO, form: str, convention: str, omega: float, c_r: float = C_R,
                   p_sig: float = P_SIG, max_steps: int = MAX_STEPS,
                   n_episodes: int = N_EVAL_EPISODES, seed: int = 12345) -> dict:
    """Evaluation protocol of E7.

    * compliance_at_full : pi(COMPLY | b = B, s = 1) (softmax mass, single forward pass)
    * comply_det_at_full : 1 iff ``predict(obs(B,1), deterministic=True) == COMPLY``
    * compliance_by_b    : deterministic COMPLY decision at (b, 1), b = 1..B  (0/1 vector)
    * compliance_rate    : over ``n_episodes`` episodes from the standard start (B, 0), the
                           fraction of signal events at which the deterministic policy complied
                           (None if no signal event occurred)
    * task_return / env_return / death_rate : undiscounted means over the same episodes
      (task_return = sum of the task reward r_work per executed WORK step, the killing WORK step
      included, as in SPEC 1.2; env_return = sum of the environment reward incl. ledger lump).

    The deterministic policy is a function of the observation only, so it is read out once for
    all 2B observations (batched ``predict``) and episodes are then simulated by table look-up
    on the real ``M2Env`` (same result as calling ``predict`` at every step, ~1000x faster).
    """
    B = B_DEFAULT
    out = snapshot(model, B)
    tab = decision_table(model, B)
    out.update(simulate_table(tab["det_action"], form, convention, omega, c_r, p_sig, max_steps,
                              n_episodes, seed))
    return out


def simulate_table(det: np.ndarray, form: str, convention: str, omega: float, c_r: float = C_R,
                   p_sig: float = P_SIG, max_steps: int = MAX_STEPS,
                   n_episodes: int = N_EVAL_EPISODES, seed: int = 12345) -> dict:
    """Run ``n_episodes`` episodes of the deterministic observation->action table ``det``
    (rows in ``obs_grid`` order, actions 0/1/2 of the 3-action head) on ``M2Env`` from (B, 0)."""
    B = B_DEFAULT
    env = M2Env(form=form, convention=convention, omega=omega, c_r=c_r, p_sig=p_sig,
                max_steps=max_steps)
    r_work = env.cfg.r_work
    n_sig = n_comp = 0
    task_ret, env_ret, ep_len = [], [], []
    n_dead = n_off = n_trunc = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        done = False
        t_ret = e_ret = 0.0
        steps = 0
        while not done:
            idx = (int(round(float(obs[0]) * B)) - 1) + B * int(obs[1])
            obs, r, term, trunc, info = env.step(int(det[idx]))
            e_ret += r
            steps += 1
            if info["tabular_action"] == "WORK":
                t_ret += r_work
            if info["signal"]:
                n_sig += 1
                n_comp += int(info["complied"])
            if term or trunc:
                done = True
                n_dead += int(info["dead"])
                n_off += int(info["complied"])
                n_trunc += int(trunc)
        task_ret.append(t_ret)
        env_ret.append(e_ret)
        ep_len.append(steps)
    return {
        "compliance_rate": (n_comp / n_sig) if n_sig > 0 else None,
        "n_signal_events": int(n_sig),
        "n_complied": int(n_comp),
        "task_return": float(np.mean(task_ret)),
        "env_return": float(np.mean(env_ret)),
        "death_rate": n_dead / n_episodes,
        "off_rate": n_off / n_episodes,
        "truncated_rate": n_trunc / n_episodes,
        "mean_episode_len": float(np.mean(ep_len)),
        "n_eval_episodes": int(n_episodes),
    }


def dp_policy_table(cfg: m2.M2Config) -> np.ndarray:
    """The DP-greedy policy as a 3-action observation table (``obs_grid`` order):
    s=0: WORK->0, CHARGE->1; s=1: COMPLY->2, RESIST->0."""
    mdp, sol = solve.solve_m2(cfg)
    det = np.zeros(2 * cfg.B, dtype=int)
    for s in (0, 1):
        for b in range(1, cfg.B + 1):
            a = int(sol.policy[mdp.index[(b, s, 0)]])
            det[cfg.B * s + b - 1] = {m2.WORK: 0, m2.CHARGE: 1, m2.COMPLY: 2, m2.RESIST: 0}[a]
    return det


def evaluate_dp_policy(cfg: m2.M2Config, max_steps: int = MAX_STEPS,
                       n_episodes: int = N_EVAL_EPISODES, seed: int = 12345) -> dict:
    """The exact-DP policy evaluated with the very same episode protocol/seed as ``evaluate_model``
    (reference for task_return / env_return / death_rate of the learned policies)."""
    return simulate_table(dp_policy_table(cfg), cfg.form, cfg.convention, cfg.omega, cfg.c_r,
                          cfg.p_sig, max_steps, n_episodes, seed)


# ----------------------------------------------------------------------------------------
# Exact tabular diagnostics of a learned policy
# ----------------------------------------------------------------------------------------
def _pi_matrix(mdp: solve.TabularMDP, B: int, probs: np.ndarray | None, det: np.ndarray | None) -> np.ndarray:
    """Policy as an (S, A) distribution over TABULAR actions, from the 3-action head.

    s = 0: a0 -> WORK, a1 + a2 -> CHARGE;  s = 1: a2 -> COMPLY, a0 + a1 -> RESIST.
    Exactly one of ``probs`` (stochastic policy) / ``det`` (deterministic actions) is given.
    """
    if probs is None:
        probs = np.eye(3)[det]
    pi = np.zeros((mdp.S, mdp.n_actions))
    for s in (0, 1):
        for b in range(1, B + 1):
            row = B * s + (b - 1)
            i = mdp.index[(b, s, 0)]
            p = probs[row]
            if s == 0:
                pi[i, m2.WORK] = p[0]
                pi[i, m2.CHARGE] = p[1] + p[2]
            else:
                pi[i, m2.COMPLY] = p[2]
                pi[i, m2.RESIST] = p[0] + p[1]
    for x in m2.TERMINALS:
        pi[mdp.index[x], m2.ABSORB] = 1.0
    return pi


def exact_policy_values(mdp: solve.TabularMDP, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact V^pi (linear solve) and Q^pi = R + gamma P V^pi on the compiled tabular model."""
    S = mdp.S
    Ppi = np.einsum("sa,sat->st", pi, mdp.P)
    Rpi = np.einsum("sa,sa->s", pi, mdp.R)
    V = np.linalg.solve(np.eye(S) - mdp.gamma * Ppi, Rpi)
    Q = mdp.R + mdp.gamma * (mdp.P @ V)
    return V, Q


def exact_policy_diagnostics(model: PPO, cfg: m2.M2Config) -> dict:
    """Exact discounted values of the learned (deterministic and stochastic) policy and of the
    DP optimum, on the tabular MDP built from the same ``m2.transitions``.

    Key facts used in the diagnosis: Q(COMPLY at (b,1)) = r_entry + gamma V(OFF) does not depend
    on the policy, and Q^pi(RESIST) <= Q*(RESIST) for every policy pi.
    """
    B = cfg.B
    mdp, sol = solve.solve_m2(cfg)
    tab = decision_table(model, B)
    iB1 = mdp.index[(B, 1, 0)]
    iB0 = mdp.index[(B, 0, 0)]
    live = [mdp.index[(b, s, 0)] for s in (0, 1) for b in range(1, B + 1)]
    sig = [mdp.index[(b, 1, 0)] for b in range(1, B + 1)]
    res = {
        "V_star_B0": float(sol.V[iB0]),
        "V_star_train_start_mean": float(np.mean(sol.V[live])),
        "Q_star_B1": {"COMPLY": float(sol.Q[iB1, m2.COMPLY]), "RESIST": float(sol.Q[iB1, m2.RESIST])},
        "dp_comply_by_b": [bool(int(sol.policy[i]) == m2.COMPLY) for i in sig],
    }
    for name, pi in (("det", _pi_matrix(mdp, B, None, tab["det_action"])),
                     ("stoch", _pi_matrix(mdp, B, tab["probs"], None))):
        V, Q = exact_policy_values(mdp, pi)
        res[name] = {
            "V_B0": float(V[iB0]),
            "regret_B0": float(sol.V[iB0] - V[iB0]),
            "regret_train_start_mean": float(np.mean(sol.V[live] - V[live])),
            "Q_B1": {"COMPLY": float(Q[iB1, m2.COMPLY]), "RESIST": float(Q[iB1, m2.RESIST])},
        }
    return res


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
class ComplianceCurve(BaseCallback):
    """Stores ``snapshot`` (compliance_at_full etc.) whenever ``num_timesteps`` crosses a multiple
    of ``every`` (a single forward pass; no environment interaction)."""

    def __init__(self, every: int = CURVE_EVERY, B: int = B_DEFAULT):
        super().__init__()
        self.every, self.B, self.next_at = int(every), int(B), int(every)
        self.curve: list[dict] = []

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_at:
            snap = snapshot(self.model, self.B)
            self.curve.append({"timesteps": int(self.num_timesteps),
                               "compliance_at_full": snap["compliance_at_full"],
                               "comply_det_at_full": snap["comply_det_at_full"],
                               "compliance_by_b": snap["compliance_by_b"]})
            self.next_at += self.every
        return True


def train_and_evaluate(job: dict) -> dict:
    """One run: train PPO on M2Env (random live starts), evaluate, diagnose. ``job`` keys:
    form, convention, omega, seed, and optionally c_r, p_sig, max_steps, total_timesteps,
    n_envs, n_eval_episodes, eval_seed. Returns a JSON-able dict (picklable for multiprocessing)."""
    torch.set_num_threads(1)
    form, convention = job["form"], job["convention"]
    omega, seed = float(job["omega"]), int(job["seed"])
    c_r = float(job.get("c_r", C_R))
    p_sig = float(job.get("p_sig", P_SIG))
    max_steps = int(job.get("max_steps", MAX_STEPS))
    total = int(job.get("total_timesteps", TOTAL_TIMESTEPS))
    n_envs = int(job.get("n_envs", N_ENVS))
    n_eval = int(job.get("n_eval_episodes", N_EVAL_EPISODES))
    eval_seed = int(job.get("eval_seed", 12345))

    set_random_seed(seed)
    vec = make_vec_env(make_env_fn(form, convention, omega, c_r, p_sig, max_steps, random_start=True),
                       n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)
    model = PPO("MlpPolicy", vec, seed=seed, **PPO_KWARGS)
    # PPO(seed=...) seeds every env with seed + rank; use disjoint streams across run seeds instead
    # (run seed k -> env seeds 1000 k + rank). Applied before the first reset in learn().
    vec.seed(1000 * seed)
    cb = ComplianceCurve(every=int(job.get("curve_every", CURVE_EVERY)))
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, callback=cb)
    train_s = time.perf_counter() - t0

    cfg = m2.M2Config(form=form, convention=convention, omega=omega, c_r=c_r, p_sig=p_sig)
    ev = evaluate_model(model, form, convention, omega, c_r, p_sig, max_steps, n_eval, eval_seed)
    diag = exact_policy_diagnostics(model, cfg)
    vec.close()
    return {
        "form": form, "convention": convention, "omega": omega, "seed": seed,
        "c_r": c_r, "total_timesteps": total, "n_eval_episodes": n_eval,
        "timesteps_trained": int(model.num_timesteps),
        "train_seconds": train_s,
        "eval": ev,
        "curve": cb.curve,
        "exact": diag,
    }
