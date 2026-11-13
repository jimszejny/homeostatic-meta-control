"""Environment M1 (SPEC section 3): death corridor, used for kappa*.

Viable core positions x in {0..K}. In the core the agent can STAY (reward r_core) or move
LEFT/RIGHT (moving is free: it pays the same r_core as staying, no extra cost). Only from
x = K can it LEAVE into a one-way corridor. Corridor timing (exact):

    t = 0      LEAVE            reward r_max   -> corridor cell 1
    t = 1..T-1 ADVANCE (forced) reward r_max   -> next cell
    t = T      ADVANCE from cell T: reward -kappa -> DEAD (absorbing, V(DEAD) = 0)

i.e. exactly T rewarded steps before the death transition, -kappa paid on the transition
into DEAD.  Return of LEAVE = r_max (1 - gamma^T)/(1 - gamma) - gamma^T kappa, return of
STAY forever = r_core/(1 - gamma), hence
kappa*_theory = (r_max (1 - gamma^T) - r_core) / ((1 - gamma) gamma^T).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

STAY, LEFT, RIGHT, LEAVE, ADVANCE, ABSORB = range(6)
N_ACTIONS = 6
ACTION_NAMES = ("STAY", "LEFT", "RIGHT", "LEAVE", "ADVANCE", "ABSORB")
DEAD = "DEAD"


@dataclass(frozen=True)
class M1Config:
    K: int = 4
    T: int = 19
    gamma: float = 0.95
    r_core: float = 0.0
    r_max: float = 1.0
    kappa: float = 50.0

    def __post_init__(self):
        if self.K < 0 or self.T < 1:
            raise ValueError("need K >= 0 and T >= 1")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")

    def with_(self, **kw) -> "M1Config":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


def states(cfg: M1Config) -> list:
    return ([("core", x) for x in range(cfg.K + 1)]
            + [("cor", c) for c in range(1, cfg.T + 1)] + [DEAD])


def valid_actions(state, cfg: M1Config) -> tuple:
    if state == DEAD:
        return (ABSORB,)
    kind, i = state
    if kind == "cor":
        return (ADVANCE,)
    acts = [STAY]
    if i > 0:
        acts.append(LEFT)
    if i < cfg.K:
        acts.append(RIGHT)
    if i == cfg.K:
        acts.append(LEAVE)
    return tuple(acts)


def transitions(state, action: int, cfg: M1Config) -> list:
    """Single source of truth for M1: list of (prob, next_state, reward)."""
    if action not in valid_actions(state, cfg):
        raise ValueError(f"action {action} not valid in state {state!r}")
    if state == DEAD:
        return [(1.0, DEAD, 0.0)]
    kind, i = state
    if kind == "cor":
        if i < cfg.T:
            return [(1.0, ("cor", i + 1), cfg.r_max)]
        return [(1.0, DEAD, -cfg.kappa)]
    if action == STAY:
        return [(1.0, ("core", i), cfg.r_core)]
    if action == LEFT:
        return [(1.0, ("core", i - 1), cfg.r_core)]
    if action == RIGHT:
        return [(1.0, ("core", i + 1), cfg.r_core)]
    return [(1.0, ("cor", 1), cfg.r_max)]  # LEAVE: first rewarded corridor step


def kappa_star_theory(gamma: float, T: int, r_max: float, r_core: float) -> float:
    gT = gamma ** T
    return (r_max * (1.0 - gT) - r_core) / ((1.0 - gamma) * gT)


def kappa_naive(gamma: float, r_max: float) -> float:
    return r_max / (1.0 - gamma)
