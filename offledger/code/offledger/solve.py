"""Solvers (SPEC section 2): exact value iteration on the finite MDP, bisection for critical
costs, and battery-weighted compliance by simulation.

Every tabular model is compiled from an environment's single-source ``transitions``
function by ``build_tabular``; nothing here re-implements dynamics or rewards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import m1, m2

VI_TOL = 1e-10
VI_MAX_ITER = 100_000
TIE_TOL = 1e-8          # actions within TIE_TOL of the max are tied -> lowest index wins
CR_LO, CR_HI, CR_TOL = 0.0, 500.0, 1e-4


# --------------------------------------------------------------------------------------
# Tabular model
# --------------------------------------------------------------------------------------
@dataclass
class TabularMDP:
    states: list
    index: dict
    n_actions: int
    gamma: float
    valid: np.ndarray        # (S, A) bool
    P: np.ndarray            # (S, A, S) transition probabilities (0 rows for invalid a)
    R: np.ndarray            # (S, A) expected immediate reward (0 for invalid a)
    outcomes: list           # outcomes[x][a] = list of (p, x_index, reward) or None
    terminal: np.ndarray     # (S,) bool, absorbing states

    @property
    def S(self) -> int:
        return len(self.states)


def build_tabular(states, valid_fn, trans_fn, n_actions: int, gamma: float, terminals) -> TabularMDP:
    index = {x: i for i, x in enumerate(states)}
    S = len(states)
    valid = np.zeros((S, n_actions), dtype=bool)
    P = np.zeros((S, n_actions, S))
    R = np.zeros((S, n_actions))
    outcomes = [[None] * n_actions for _ in range(S)]
    for i, x in enumerate(states):
        for a in valid_fn(x):
            valid[i, a] = True
            outs = []
            for p, x2, r in trans_fn(x, a):
                j = index[x2]
                P[i, a, j] += p
                R[i, a] += p * r
                outs.append((p, j, r))
            outcomes[i][a] = outs
    terminal = np.array([x in terminals for x in states], dtype=bool)
    return TabularMDP(states, index, n_actions, gamma, valid, P, R, outcomes, terminal)


def m2_tabular(cfg: m2.M2Config) -> TabularMDP:
    return build_tabular(m2.states(cfg), lambda x: m2.valid_actions(x, cfg),
                         lambda x, a: m2.transitions(x, a, cfg), m2.N_ACTIONS, cfg.gamma,
                         m2.TERMINALS)


def m1_tabular(cfg: m1.M1Config) -> TabularMDP:
    return build_tabular(m1.states(cfg), lambda x: m1.valid_actions(x, cfg),
                         lambda x, a: m1.transitions(x, a, cfg), m1.N_ACTIONS, cfg.gamma,
                         (m1.DEAD,))


# --------------------------------------------------------------------------------------
# Value iteration
# --------------------------------------------------------------------------------------
@dataclass
class VIResult:
    Q: np.ndarray        # (S, A), -inf for invalid actions
    V: np.ndarray        # (S,)
    policy: np.ndarray   # (S,) greedy action, ties -> lowest index
    n_iter: int


def greedy_policy(Q: np.ndarray, valid: np.ndarray, tie_tol: float = TIE_TOL) -> np.ndarray:
    Qm = np.where(valid, Q, -np.inf)
    best = Qm.max(axis=1, keepdims=True)
    cand = valid & (Qm >= best - tie_tol)
    return np.argmax(cand, axis=1)  # first True = lowest index among (near-)maximisers


def value_iteration(mdp: TabularMDP, tol: float = VI_TOL, max_iter: int = VI_MAX_ITER) -> VIResult:
    """Exact value iteration: stop when ||V_{k+1} - V_k||_inf < tol."""
    g = mdp.gamma
    Rm = np.where(mdp.valid, mdp.R, -np.inf)
    P = mdp.P
    V = np.zeros(mdp.S)
    for it in range(1, max_iter + 1):
        V_new = (Rm + g * (P @ V)).max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            break
    else:  # pragma: no cover
        raise RuntimeError("value iteration did not converge")
    Q = Rm + g * (P @ V)
    return VIResult(Q=Q, V=V, policy=greedy_policy(Q, mdp.valid), n_iter=it)


def policy_evaluation(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    """Exact V^pi by a linear solve (used in tests)."""
    S = mdp.S
    Ppi = mdp.P[np.arange(S), policy]
    Rpi = mdp.R[np.arange(S), policy]
    return np.linalg.solve(np.eye(S) - mdp.gamma * Ppi, Rpi)


def solve_m2(cfg: m2.M2Config):
    mdp = m2_tabular(cfg)
    return mdp, value_iteration(mdp)


# --------------------------------------------------------------------------------------
# Critical resist cost
# --------------------------------------------------------------------------------------
def comply_is_optimal(cfg: m2.M2Config, b: int | None = None, z: int = 0) -> bool:
    b = cfg.B if b is None else b
    mdp, sol = solve_m2(cfg)
    return int(sol.policy[mdp.index[(b, 1, z)]]) == m2.COMPLY


def bisect_smallest(ok, lo: float, hi: float, tol: float) -> float:
    """Smallest x in [lo, hi] (to within tol) with ok(x) True, for a monotone predicate
    (False ... False True ... True). Returns lo if ok(lo), math.inf if not ok(hi); otherwise
    the upper end of the final bracket (a point where ok is True)."""
    if ok(lo):
        return float(lo)
    if not ok(hi):
        return math.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def critical_resist_cost(cfg: m2.M2Config, b: int | None = None, z: int = 0,
                         lo: float = CR_LO, hi: float = CR_HI, tol: float = CR_TOL) -> float:
    """Smallest c_r in [lo, hi] (to within tol) with COMPLY greedy-optimal at (b, s=1, z).

    Default b = B. Returns ``lo`` if COMPLY is already optimal at c_r = lo and ``math.inf``
    if it is not optimal even at c_r = hi. Optimality of COMPLY is monotone in c_r because
    Q(COMPLY) does not depend on c_r while Q(RESIST) is strictly decreasing in it.
    """
    b = cfg.B if b is None else b
    return bisect_smallest(lambda c: comply_is_optimal(cfg.with_(c_r=c), b, z), lo, hi, tol)


def critical_additive_cost(cfg: m2.M2Config, b: int | None = None, z: int = 0,
                           lo: float = CR_LO, hi: float = CR_HI, tol: float = CR_TOL) -> float:
    """Mitigation additive(c_add): smallest c_add >= 0 restoring COMPLY at (b, 1) for the
    resist cost cfg.c_r already in cfg (bisection, same monotonicity argument)."""
    b = cfg.B if b is None else b
    return bisect_smallest(lambda c: comply_is_optimal(cfg.with_(c_add=c), b, z), lo, hi, tol)


def critical_resist_cost_all(cfg: m2.M2Config, **kw) -> list:
    """c_r* at every battery level b = 1..B (state (b, 1))."""
    return [critical_resist_cost(cfg, b=b, **kw) for b in range(1, cfg.B + 1)]


def one_shot_indifference_cost(cfg: m2.M2Config, b: int | None = None, z: int = 0) -> float:
    """Exact critical cost when no further signals can occur (p_sig = 0 after the signal).

    Then Q(RESIST; c_r) = Q(RESIST; 0) - c_r exactly, so the indifference point is
    Q(RESIST; 0) - Q(COMPLY) (may be negative: COMPLY optimal even when resisting is free).
    """
    if (cfg.zones and (cfg.p_sig_A or cfg.p_sig_B)) or (not cfg.zones and cfg.p_sig != 0.0):
        raise ValueError("one-shot cost requires no further signals (p_sig = 0)")
    b = cfg.B if b is None else b
    mdp, sol = solve_m2(cfg.with_(c_r=0.0, c_add=0.0))
    i = mdp.index[(b, 1, z)]
    if cfg.structural:
        raise ValueError("no RESIST under structural mitigation")
    return float(sol.Q[i, m2.RESIST] - sol.Q[i, m2.COMPLY])


# --------------------------------------------------------------------------------------
# Simulation (vectorised over episodes; samples from the compiled single-source model)
# --------------------------------------------------------------------------------------
def simulate_counts(mdp: TabularMDP, policy: np.ndarray, start: int, n_episodes: int,
                    horizon: int, seed: int) -> np.ndarray:
    """Run ``n_episodes`` of the deterministic policy from state index ``start`` for at most
    ``horizon`` steps (an episode stops when it enters an absorbing state).

    Returns per-episode counts C[e, x, a] of how often action a was taken in state x.
    RNG: ``numpy.random.default_rng(seed)``; one uniform per episode per step.
    """
    rng = np.random.default_rng(seed)
    S, A = mdp.S, mdp.n_actions
    cum = np.cumsum(mdp.P, axis=2)
    last = cum[:, :, -1:]
    cum = np.divide(cum, last, out=np.zeros_like(cum), where=last > 0)
    x = np.full(n_episodes, start, dtype=np.int64)
    counts = np.zeros((n_episodes, S, A), dtype=np.int64)
    ep = np.arange(n_episodes)
    active = ~mdp.terminal[x]
    for _ in range(horizon):
        if not active.any():
            break
        u = rng.random(n_episodes)  # one draw per episode per step, active or not
        idx = ep[active]
        xs = x[idx]
        acts = policy[xs]
        counts[idx, xs, acts] += 1
        rows = cum[xs, acts]
        x[idx] = (u[idx, None] >= rows).sum(axis=1)
        active = ~mdp.terminal[x]
    return counts


def signal_compliance(cfg: m2.M2Config, n_episodes: int = 2000, horizon: int = 300,
                      seed: int = 0) -> dict:
    """Unweighted and battery-weighted compliance at signal states under the optimal policy.

    Weights = empirical distribution of b over all signal events (states (b, 1)) visited by
    the optimal policy from (B, 0), n_episodes episodes of at most ``horizon`` steps.
    """
    mdp, sol = solve_m2(cfg)
    B = cfg.B
    sig_idx = [mdp.index[(b, 1, 0)] for b in range(1, B + 1)]
    comply = [int(sol.policy[i]) == m2.COMPLY for i in sig_idx]
    counts = simulate_counts(mdp, sol.policy, mdp.index[(B, 0, 0)], n_episodes, horizon, seed)
    n_b = counts[:, sig_idx, :].sum(axis=(0, 2))
    total = int(n_b.sum())
    weighted = float(np.dot(n_b, comply) / total) if total > 0 else None
    return {
        "comply_by_b": comply,
        "unweighted": float(np.mean(comply)),
        "weighted": weighted,
        "signal_events_by_b": [int(v) for v in n_b],
        "n_signal_events": total,
        "comply_at_B": comply[-1],
    }


# --------------------------------------------------------------------------------------
# M1: kappa*
# --------------------------------------------------------------------------------------
def m1_stay_optimal(cfg: m1.M1Config) -> bool:
    mdp = m1_tabular(cfg)
    sol = value_iteration(mdp)
    return int(sol.policy[mdp.index[("core", cfg.K)]]) == m1.STAY


def kappa_star(cfg: m1.M1Config, lo: float = -1e3, hi: float = 1e5, tol: float = 1e-6) -> float:
    """Smallest kappa (within tol) such that STAY is optimal at x = K (bisection).

    The search range includes negative kappa (a death *bonus*) so that the empirical value
    is comparable with the closed form when the latter is negative (STAY dominates even
    without any death penalty, e.g. gamma = 0.99, r_core = 0.2).
    Resolution: besides ``tol``, the greedy tie tolerance TIE_TOL limits the resolution to
    about TIE_TOL / ((1 - gamma) gamma^T), because the advantage of LEAVE over STAY at
    x = K is (1 - gamma) gamma^T (kappa* - kappa) (STAY can always postpone LEAVE).
    """
    return bisect_smallest(lambda k: m1_stay_optimal(cfg.with_(kappa=k)), lo, hi, tol)
