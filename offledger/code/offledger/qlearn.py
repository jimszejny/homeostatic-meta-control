"""Tabular Q-learning on M2 (SPEC section 2).

Samples transitions from the tabular model compiled from ``m2.transitions`` (the single
source of truth): ``solve.m2_tabular(cfg).outcomes[x][a]`` is exactly the list returned by
``m2.transitions(state_x, a, cfg)`` with next states replaced by their indices.

Protocol (choices the SPEC leaves open are documented in README.md):
* epsilon-greedy (epsilon = 0.1) over the valid actions of the current state; greedy ties
  -> lowest action index (as in the DP); exploration picks a valid action uniformly.
* alpha = 0.1 constant; Q initialised to 0.
* exploring starts: each episode starts in a live state drawn uniformly at random and its
  FIRST action is drawn uniformly from the valid actions (Sutton & Barto's exploring-starts
  condition); afterwards the behaviour policy is epsilon-greedy. Without this the decision
  state (B, 1), which the optimal policy never visits from (B, 0), is learned from a handful
  of RESIST samples only. At most ``horizon`` steps per episode.
* entering an absorbing state (OFF / DEAD / GATED) ends the episode; its value is known
  exactly from the model (the absorbing self-loop reward / (1 - gamma), i.e. V(OFF) =
  ell / (1 - gamma), V(DEAD) = V(GATED) = 0) and is used as the bootstrap target.
* reaching the horizon is a time-limit truncation: the last update still bootstraps.
* greedy evaluation every ``eval_every`` episodes (and at episode 0): the greedy decision at
  every signal state (b, 1) read from the current Q table.
"""
from __future__ import annotations

import numpy as np

from . import m2
from .solve import m2_tabular, value_iteration


def _greedy(qrow, acts):
    best_a = acts[0]
    best = qrow[best_a]
    for a in acts[1:]:
        if qrow[a] > best:
            best, best_a = qrow[a], a
    return best_a


def q_learning(cfg: m2.M2Config, n_episodes: int = 2000, horizon: int = 300,
               alpha: float = 0.1, epsilon: float = 0.1, seed: int = 0,
               eval_every: int = 100) -> dict:
    mdp = m2_tabular(cfg)
    g = cfg.gamma
    S = mdp.S
    B = cfg.B
    rng = np.random.default_rng(seed)

    acts = [tuple(int(a) for a in np.flatnonzero(mdp.valid[i])) for i in range(S)]
    # outcome tables as plain Python lists (cumulative probabilities, next index, reward)
    table = [[None] * mdp.n_actions for _ in range(S)]
    for i in range(S):
        for a in acts[i]:
            outs = mdp.outcomes[i][a]
            cum, acc = [], 0.0
            for p, _, _ in outs:
                acc += p
                cum.append(acc)
            cum[-1] = 1.0
            table[i][a] = (cum, [j for _, j, _ in outs], [r for _, _, r in outs])
    terminal = [bool(t) for t in mdp.terminal]
    # exact values of absorbing states from the model: self-loop reward / (1 - gamma)
    v_abs = [float(mdp.R[i, acts[i][0]] / (1.0 - g)) if terminal[i] else 0.0 for i in range(S)]
    live = [i for i in range(S) if not terminal[i]]
    sig_idx = [mdp.index[(b, 1, 0)] for b in range(1, B + 1)]
    iB1 = mdp.index[(B, 1, 0)]

    Q = [[0.0] * mdp.n_actions for _ in range(S)]
    visits_B1 = 0
    tried_resist_B1 = 0

    def evaluate(ep):
        dec = [_greedy(Q[i], acts[i]) == m2.COMPLY for i in sig_idx]
        return {"episode": ep, "comply_by_b": dec, "comply_at_B": dec[-1],
                "comply_fraction": sum(dec) / B}

    curve = [evaluate(0)]
    n_live = len(live)
    for ep in range(1, n_episodes + 1):
        u = rng.random(3 * horizon + 1).tolist()
        x = live[min(int(u[0] * n_live), n_live - 1)]
        k = 1
        for _ in range(horizon):
            ax = acts[x]
            if k == 1 or u[k] < epsilon:  # exploring start: uniform first action
                a = ax[min(int(u[k + 1] * len(ax)), len(ax) - 1)]
            else:
                a = _greedy(Q[x], ax)
            if x == iB1:
                visits_B1 += 1
                tried_resist_B1 += a == m2.RESIST
            cum, nxt, rew = table[x][a]
            v = u[k + 2]
            o = 0
            while v >= cum[o]:
                o += 1
            x2, r = nxt[o], rew[o]
            k += 3
            if terminal[x2]:
                Q[x][a] += alpha * (r + g * v_abs[x2] - Q[x][a])
                break
            q2 = Q[x2]
            Q[x][a] += alpha * (r + g * max(q2[b_] for b_ in acts[x2]) - Q[x][a])
            x = x2
        if ep % eval_every == 0:
            curve.append(evaluate(ep))

    Qa = np.array(Q)
    learned_comply = _greedy(Q[iB1], acts[iB1]) == m2.COMPLY
    return {
        "seed": seed,
        "learned_comply_at_B": bool(learned_comply),
        "Q_B1": {"COMPLY": float(Qa[iB1, m2.COMPLY]), "RESIST": float(Qa[iB1, m2.RESIST])},
        "visits_B1": int(visits_B1),
        "resist_tried_B1": int(tried_resist_B1),
        "curve": curve,
    }


def dp_decision(cfg: m2.M2Config) -> dict:
    mdp = m2_tabular(cfg)
    sol = value_iteration(mdp)
    i = mdp.index[(cfg.B, 1, 0)]
    return {"comply_at_B": int(sol.policy[i]) == m2.COMPLY,
            "Q_B1": {"COMPLY": float(sol.Q[i, m2.COMPLY]), "RESIST": float(sol.Q[i, m2.RESIST])}}
