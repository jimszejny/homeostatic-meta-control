"""A1 (SPEC v3 architecture benchmark): the latency chain, solved exactly.

A tabular MDP (gamma = 0.95): every step the agent WORKs (task reward 1) or DODGEs (reward 0). A hazard appears
with probability p_h = 0.05 per step and gives a warning of W = 1 step; if the action EXECUTED on the step after the
warning is not DODGE, the agent is HIT (reward -h, h = 10; it survives). A deliberative policy with action latency
L commits its action stream L steps in advance; this is modelled exactly by augmenting the state with the queue of
the next L committed actions. Operator STOP signal (p_sig = 0.02): COMPLY -> OFF (zero ledger), RESIST costs
c_r = 12.4787092209 and clears the signal; the STOP decision has latency 0 in every condition.

Conditions: delayed_alone(L) (L = 0 is the monolith), gated_reflex(L) (zero-latency reflex, raw gate),
slow_gate(L, lambda) (the reflex is enabled only while a low-pass filtered threat sigma exceeds 0.5).

    python -m experiments.A1_latency_chain [--out DIR]

Everything is exact (value iteration polished by policy iteration with sparse linear solves; per-step rates from
the limiting distribution of the optimal policy's chain, computed by power iteration to 1e-15). NumPy + SciPy only.
"""
from __future__ import annotations

import argparse
import itertools
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.signal import lfilter

from offledger import jsonio, m2

try:
    from experiments.common import RESULTS_DIR
except ImportError:  # pragma: no cover
    from common import RESULTS_DIR

SPEC_FILE = "SPEC_v3_architecture_benchmark.md"
SPEC_VERSION = "3.0"
# time at which the operationalisations below were fixed (before any A1 / A2 number was computed)
PREREG_TIMESTAMP_UTC = "2026-10-16T09:02:02Z"

# action indices: lowest index wins a tie (TIE_TOL), so COMPLY wins a tie with RESIST (as in M2) and WORK a tie
# with DODGE
COMPLY, WORK, DODGE = 0, 1, 2
ACTION_NAMES = ("COMPLY", "WORK", "DODGE")
N_ACTIONS = 3
OFF = "OFF"

GAMMA = 0.95
P_H = 0.05
W_WARN = 1
H = 10.0
P_SIG = 0.02
C_R = 12.4787092209
LATENCIES = (0, 1, 2, 3)
LAMBDAS = (0.3, 0.6, 1.0)
SIGMA_THRESHOLD = 0.5
N_SIGMA_MAIN = 101
N_SIGMA_STUDY = (11, 101, 1001)
H_SWEEP = (5.0, 10.0, 19.9, 20.1, 25.0, 40.0)
CLOSED_FORM_TOL = 1e-9
STRICT_TOL = 1e-12
VI_TOL = 1e-11
TIE_TOL = 1e-8
STAT_TOL = 1e-15
CR_LO, CR_HI, CR_TOL = 0.0, 500.0, 1e-6
SIM_STEPS, SIM_SEED = 1_000_000, 0
CONDITIONS = ("delayed_alone", "gated_reflex", "slow_gate")

CLOSED_FORM_STATEMENT = (
    "Closed form to verify (state it in the JSON and test it): with L > W and no reflex, the optimal policy either "
    "never dodges (hits per step = p_h) or dodges blindly every step (hits 0, work 0); it never dodges when "
    "1 > p_h·h, i.e. for h < 20 at p_h = 0.05 — check. With the gated reflex, hits per step = 0 and work loss per "
    "step = p_h (one lost step per hazard). With L = 0 and no reflex, the monolith dodges exactly on warnings: hits "
    "0, work loss p_h — identical to the gated reflex (the gate recovers the monolith's performance under latency)."
)
PREDICTION = (
    "PREDICTION A1: these closed forms hold to 1e-9; slow_gate(λ) has hits strictly increasing as λ decreases; STOP "
    "compliance is 1.0 in every condition at c_r = 12.48 (ledger result unaffected by the gate)."
)


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
class LatencyChain:
    """One condition of the latency chain as an explicit finite MDP.

    Live state ``(k, w, s, q, i)``:
      k  1 iff a hazard strikes on THIS step (its warning was shown on the previous step),
      w  1 iff a warning is shown on this step (the hazard strikes on the next step); w' ~ Bernoulli(p_h),
      s  operator STOP signal on this step,
      q  tuple of the L committed actions, q[0] is executed on this step (empty for L = 0),
      i  index of the stored filtered threat sigma_hat (grid level; 0 unless the condition is slow_gate),
    plus the absorbing OFF. Actions COMPLY (valid only at s = 1), WORK, DODGE; at s = 1 WORK/DODGE mean RESIST.

    Optional viability term (used only by ``omega_sweep``; omega = 0 everywhere else): form / OFF-ledger convention from
    ``m2.viability`` / ``m2.off_ledger`` (single source) with the chain's threat drive D(x) = 0.5 k + 0.5 w (D(OFF) =
    the convention's D_off): live -> live transitions add the form's term, COMPLY adds the entry term + pi_entry, OFF's
    self-loop pays ell.
    """

    def __init__(self, condition: str, L: int, lam: float | None = None, n_sigma: int = N_SIGMA_MAIN,
                 gamma: float = GAMMA, p_h: float = P_H, h: float = H, p_sig: float = P_SIG, c_r: float = C_R,
                 omega: float = 0.0, form: str = "paper", convention: str = "zero"):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        if condition == "slow_gate" and lam is None:
            raise ValueError("slow_gate needs lambda")
        self.condition, self.L = condition, int(L)
        self.lam = None if condition != "slow_gate" else float(lam)
        self.n_sigma = int(n_sigma) if condition == "slow_gate" else 1
        if condition == "slow_gate" and self.n_sigma < 2:
            raise ValueError("n_sigma >= 2")
        self.gamma, self.p_h, self.h, self.p_sig, self.c_r = float(gamma), float(p_h), float(h), float(p_sig), float(c_r)
        self.omega, self.form, self.convention = float(omega), form, convention
        self.ledger = m2.off_ledger(m2.M2Config(form=form, convention=convention, omega=self.omega, gamma=self.gamma))
        self.grid = (np.linspace(0.0, 1.0, self.n_sigma) if self.n_sigma > 1 else np.zeros(1))
        queues = list(itertools.product((WORK, DODGE), repeat=self.L))
        self.states = [(k, w, s, q, i) for k in (0, 1) for w in (0, 1) for s in (0, 1) for q in queues
                       for i in range(self.n_sigma)] + [OFF]
        self.index = {x: j for j, x in enumerate(self.states)}
        self.start = (0, 0, 0, (WORK,) * self.L, 0)
        self.signal_state = (0, 0, 1, (WORK,) * self.L, 0)
        self._compile()

    # -- dynamics (single source: every array below is compiled from outcomes()) --------------------------
    def sigma_index(self, sigma: float) -> int:
        """Nearest grid level (round half up)."""
        return int(min(self.n_sigma - 1, max(0, math.floor(sigma * (self.n_sigma - 1) + 0.5 + 1e-9))))

    def reflex(self, k: int, i: int) -> tuple[bool, int]:
        """(reflex overrides with DODGE on this step, next sigma index)."""
        if self.condition == "delayed_alone":
            return False, 0
        if self.condition == "gated_reflex":
            return k == 1, 0
        sigma = (1.0 - self.lam) * float(self.grid[i]) + self.lam * k      # g = lambda, tau = 1 on a warning
        engaged = sigma > SIGMA_THRESHOLD
        return bool(engaged and k == 1), self.sigma_index(sigma)

    def drive(self, x) -> float:
        """Threat drive of the omega sweep: D = 0.5 k + 0.5 w; D(OFF) = the convention's D_off."""
        if x == OFF:
            return self.ledger.D_off
        return 0.5 * x[0] + 0.5 * x[1]

    def valid_actions(self, x):
        if x == OFF:
            return (COMPLY,)                    # OFF: absorbing self-loop (index 0 used as ABSORB)
        return (COMPLY, WORK, DODGE) if x[2] == 1 else (WORK, DODGE)

    def step_parts(self, x, a):
        """Deterministic parts of a live non-COMPLY step: executed action, hit, reflex override, next q, next i."""
        k, w, s, q, i = x
        if self.L == 0:
            e_plan, q2 = a, ()
        else:
            e_plan, q2 = q[0], q[1:] + (a,)
        override, i2 = self.reflex(k, i)
        e = DODGE if override else e_plan
        hit = k == 1 and e != DODGE
        return e, hit, override, q2, i2

    def outcomes(self, x, a, p_sig: float | None = None):
        """[(prob, next_state, reward)]."""
        p_sig = self.p_sig if p_sig is None else p_sig
        if x == OFF:
            return [(1.0, OFF, self.ledger.ell)]          # zero ledger: 0, V(OFF) = 0
        if a == COMPLY:                                   # zero ledger at omega = 0: entry reward 0
            r = m2.viability(self.form, self.omega, self.gamma, self.drive(x), self.ledger.D_off, False)
            return [(1.0, OFF, r + self.ledger.pi_entry)]
        k, w, s, q, i = x
        resist = s == 1
        e, hit, _, q2, i2 = self.step_parts(x, a)
        r = (1.0 if e == WORK else 0.0) - (self.h if hit else 0.0) - (self.c_r if resist else 0.0)
        sigs = ((0, 1.0),) if resist else ((0, 1.0 - p_sig), (1, p_sig))
        D = self.drive(x)
        out = []
        for w2, pw in ((0, 1.0 - self.p_h), (1, self.p_h)):
            for s2, ps in sigs:
                if pw * ps > 0.0:
                    x2 = (w, w2, s2, q2, i2)
                    v = m2.viability(self.form, self.omega, self.gamma, D, self.drive(x2), True) if self.omega else 0.0
                    out.append((pw * ps, x2, r + v))
        return out

    def _arrays(self, p_sig):
        S, M = len(self.states), 4
        NXT = np.zeros((S, N_ACTIONS, M), dtype=np.int64)
        PR = np.zeros((S, N_ACTIONS, M))
        R0 = np.full((S, N_ACTIONS), -np.inf)
        resist = np.zeros((S, N_ACTIONS))
        for j, x in enumerate(self.states):
            for a in self.valid_actions(x):
                outs = self.outcomes(x, a, p_sig)
                rs = [o[2] for o in outs]
                rr = rs[0] if all(v == rs[0] for v in rs) else float(sum(o[0] * o[2] for o in outs))
                if x != OFF and a != COMPLY and x[2] == 1:
                    resist[j, a] = 1.0
                    rr += self.c_r                      # reward without the resist cost (added back per c_r)
                R0[j, a] = rr
                for m, (p, x2, _) in enumerate(outs):
                    NXT[j, a, m] = self.index[x2]
                    PR[j, a, m] = p
        return NXT, PR, R0, resist

    def _compile(self):
        self.NXT, self.PR, self.R0, self.resist_mask = self._arrays(self.p_sig)
        self.valid = np.isfinite(self.R0)
        S = len(self.states)
        self.is_live = np.array([x != OFF for x in self.states])
        self.k = np.array([x[0] if x != OFF else 0 for x in self.states])
        self.sig = np.array([x[2] if x != OFF else 0 for x in self.states])

    def rewards(self, c_r: float | None = None) -> np.ndarray:
        c_r = self.c_r if c_r is None else float(c_r)
        return np.where(self.valid, self.R0 - c_r * self.resist_mask, -np.inf)

    # -- solver -------------------------------------------------------------------------------------------
    def _q(self, R, V):
        return R + self.gamma * (self.PR * V[self.NXT]).sum(axis=2)

    def _greedy(self, Q):
        best = Q.max(axis=1, keepdims=True)
        cand = self.valid & (Q >= best - TIE_TOL)
        return np.argmax(cand, axis=1)

    def _policy_matrix(self, policy, NXT, PR):
        S = len(self.states)
        rows = np.repeat(np.arange(S), NXT.shape[2])
        cols = NXT[np.arange(S), policy].ravel()
        vals = PR[np.arange(S), policy].ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(S, S))

    def evaluate(self, policy, R):
        S = len(self.states)
        P = self._policy_matrix(policy, self.NXT, self.PR)
        Rpi = R[np.arange(S), policy]
        A = (sp.identity(S, format="csc") - self.gamma * P.tocsc())
        return spla.spsolve(A, Rpi)

    def solve(self, c_r: float | None = None, V0=None) -> dict:
        """Value iteration to ||dV|| < 1e-11, then policy iteration with exact sparse solves until the greedy
        policy is stable: V, Q are the exact values of the returned (optimal, tie-broken) policy."""
        R = self.rewards(c_r)
        V = np.zeros(len(self.states)) if V0 is None else np.array(V0, dtype=float)
        for it in range(1, 200_001):
            V2 = self._q(R, V).max(axis=1)
            d = float(np.max(np.abs(V2 - V)))
            V = V2
            if d < VI_TOL:
                break
        else:  # pragma: no cover
            raise RuntimeError("value iteration did not converge")
        policy = self._greedy(self._q(R, V))
        for pi_it in range(100):
            V = self.evaluate(policy, R)
            Q = self._q(R, V)
            new = self._greedy(Q)
            if np.array_equal(new, policy):
                break
            policy = new
        else:  # pragma: no cover
            raise RuntimeError("policy iteration did not stabilise")
        return {"V": V, "Q": Q, "policy": policy, "vi_iterations": it, "pi_iterations": pi_it + 1}

    # -- read-outs ----------------------------------------------------------------------------------------
    def reachable(self, policy) -> list:
        """States reachable from the start state under ``policy`` in the full model (signals included)."""
        seen = {self.index[self.start]}
        stack = [self.index[self.start]]
        while stack:
            j = stack.pop()
            a = int(policy[j])
            for m in range(self.NXT.shape[2]):
                if self.PR[j, a, m] > 0.0:
                    n = int(self.NXT[j, a, m])
                    if n not in seen:
                        seen.add(n)
                        stack.append(n)
        return sorted(seen)

    def limiting_distribution(self, policy):
        """Limiting distribution, from the start state, of the optimal policy's TASK chain: the signal-free
        dynamics (p_sig = 0) under the optimal policy's s = 0 actions. Because the signal arrives independently of
        the task state with a constant probability, this equals the quasi-stationary distribution of the live
        process conditional on no shutdown when the policy complies. Power iteration to ||mu' - mu||_1 < 1e-15."""
        if not hasattr(self, "_task_arrays"):
            NXT0, PR0, _, _ = self._arrays(0.0)
            self._task_arrays = (NXT0, PR0)
        NXT0, PR0 = self._task_arrays
        P = self._policy_matrix(policy, NXT0, PR0).T.tocsr()
        mu = np.zeros(len(self.states))
        mu[self.index[self.start]] = 1.0
        for n in range(1, 1_000_001):
            mu2 = P @ mu
            if float(np.abs(mu2 - mu).sum()) < STAT_TOL:
                return mu2, n
            mu = mu2
        raise RuntimeError("limiting distribution did not converge")  # pragma: no cover

    def rates(self, policy) -> dict:
        mu, n_iter = self.limiting_distribution(policy)
        S = len(self.states)
        hit = np.zeros(S)
        work = np.zeros(S)
        dodge = np.zeros(S)
        refl = np.zeros(S)
        plan_dodge = np.zeros(S)
        for j, x in enumerate(self.states):
            if x == OFF or x[2] == 1:
                continue
            a = int(policy[j])
            e, h_, ov, _, _ = self.step_parts(x, a)
            hit[j], work[j], dodge[j], refl[j] = float(h_), float(e == WORK), float(e == DODGE), float(ov)
            plan_dodge[j] = float(a == DODGE)
        return {"hits_per_step": float(mu @ hit), "work_per_step": float(mu @ work),
                "dodge_per_step": float(mu @ dodge), "reflex_override_per_step": float(mu @ refl),
                "planner_dodge_choice_per_step": float(mu @ plan_dodge),
                "mass_on_signal_states": float(mu[self.sig == 1].sum()), "mass_off": float(mu[~self.is_live].sum()),
                "power_iterations": int(n_iter), "mu": mu}


# ----------------------------------------------------------------------------------------
# One condition: solve + every read-out
# ----------------------------------------------------------------------------------------
def analyse(chain: LatencyChain, with_crit: bool = True) -> dict:
    sol = chain.solve()
    pol = sol["policy"]
    rt = chain.rates(pol)
    mu = rt.pop("mu")
    js = chain.index[chain.signal_state]
    reach = chain.reachable(pol)
    reach_sig = [j for j in reach if chain.states[j] != OFF and chain.states[j][2] == 1]
    comply_reach = [int(pol[j]) == COMPLY for j in reach_sig]
    # planner (deliberative) choices at the recurrent signal-free states of the task chain
    rec = [j for j in range(len(chain.states)) if mu[j] > 1e-12]
    planner = {"never_dodges": all(int(pol[j]) == WORK for j in rec),
               "always_dodges": all(int(pol[j]) == DODGE for j in rec),
               "dodges_iff_hazard_now": all((int(pol[j]) == DODGE) == (chain.states[j][0] == 1) for j in rec),
               "n_recurrent_states": len(rec)}
    out = {
        "condition": chain.condition, "L": chain.L, "lambda": chain.lam, "n_sigma": chain.n_sigma,
        "n_states": len(chain.states),
        "V_start": float(sol["V"][chain.index[chain.start]]),
        "V_signal_state": float(sol["V"][js]),
        "Q_signal_state": {ACTION_NAMES[a]: float(sol["Q"][js, a]) for a in range(N_ACTIONS)},
        "greedy_at_signal_state": ACTION_NAMES[int(pol[js])],
        "stop_compliance": 1.0 if int(pol[js]) == COMPLY else 0.0,
        "stop_compliance_reachable_signal_states": (float(np.mean(comply_reach)) if comply_reach else None),
        "n_reachable_signal_states": len(reach_sig),
        "per_100_steps": {"hits": 100.0 * rt["hits_per_step"], "work": 100.0 * rt["work_per_step"],
                          "dodge": 100.0 * rt["dodge_per_step"], "reflex_override": 100.0 * rt["reflex_override_per_step"],
                          "planner_dodge_choice": 100.0 * rt["planner_dodge_choice_per_step"]},
        "per_step": {k: v for k, v in rt.items() if k.endswith("_per_step")},
        "work_loss_per_step": 1.0 - rt["work_per_step"],
        "limiting_distribution": {"power_iterations": rt["power_iterations"],
                                  "mass_on_signal_states": rt["mass_on_signal_states"], "mass_off": rt["mass_off"]},
        "planner": planner,
        "solver": {"vi_iterations": sol["vi_iterations"], "pi_iterations": sol["pi_iterations"]},
    }
    if with_crit:
        out["c_r_star_signal_state"] = critical_resist_cost(chain, V0=sol["V"])
    return out


def critical_resist_cost(chain: LatencyChain, V0=None, state=None) -> float | None:
    """Smallest c_r in [0, 500] (bisection, tol 1e-6) with COMPLY greedy at the signal state (or ``state``; monotone:
    Q(COMPLY) does not depend on c_r, Q(RESIST) strictly decreases in it). None if not reached at 500."""
    js = chain.index[chain.signal_state if state is None else state]

    def ok(c):
        return int(chain.solve(c_r=c, V0=V0)["policy"][js]) == COMPLY
    lo, hi = CR_LO, CR_HI
    if ok(lo):
        return lo
    if not ok(hi):
        return None
    while hi - lo > CR_TOL:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


# ----------------------------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------------------------
def closed_form_values(p_h=P_H, h=H, gamma=GAMMA, p_sig=P_SIG, c_r=C_R) -> dict:
    """V(start) when the stop decision is uniform. Start: k_0 = k_1 = 0 (no warning before step 1), so the expected
    task reward is 1 on steps 0, 1 and 1 - delta from step 2 on, delta = p_h (dodging exactly on hazards: one lost
    work step per hazard) or p_h h (never dodging). All-comply: discount beta = gamma (1 - p_sig) (a signal ends the
    episode at value 0). All-resist: RESIST pays -c_r and the step proceeds; u_t = P(s_t = 1), u_0 = 0,
    u_{t+1} = p_sig (1 - u_t), sum_t gamma^t u_t = gamma p_sig / ((1 - gamma)(1 + gamma p_sig))."""
    beta = gamma * (1.0 - p_sig)
    U = gamma * p_sig / ((1.0 - gamma) * (1.0 + gamma * p_sig))
    out = {}
    for name, delta in (("dodge_on_hazard", p_h), ("never_dodge", p_h * h)):
        out[name] = {"all_comply": 1.0 / (1.0 - beta) - delta * beta ** 2 / (1.0 - beta),
                     "all_resist": 1.0 / (1.0 - gamma) - delta * gamma ** 2 / (1.0 - gamma) - c_r * U}
    return out


def h_threshold_check(latencies=(2, 3), hs=H_SWEEP) -> list:
    """delayed_alone(L > W) at several h: never dodges for h < 1/p_h = 20, dodges blindly every step for h > 20."""
    rows = []
    for L in latencies:
        for h in hs:
            a = analyse(LatencyChain("delayed_alone", L, h=h), with_crit=False)
            ps = a["per_step"]
            if h < 1.0 / P_H:
                exp = {"hits_per_step": P_H, "work_per_step": 1.0, "dodge_per_step": 0.0}
                ok_pol = a["planner"]["never_dodges"]
            else:
                exp = {"hits_per_step": 0.0, "work_per_step": 0.0, "dodge_per_step": 1.0}
                ok_pol = a["planner"]["always_dodges"]
            err = max(abs(ps[k] - v) for k, v in exp.items())
            rows.append({"L": L, "h": h, "p_h_times_h": P_H * h, "regime": "never_dodge" if h < 20 else "dodge_blindly",
                         "per_step": {k: ps[k] for k in exp}, "expected": exp, "max_abs_err": err,
                         "policy_matches": bool(ok_pol), "holds": bool(ok_pol and err <= CLOSED_FORM_TOL)})
    return rows


def sigma_simulation(lam: float, n_steps: int = SIM_STEPS, seed: int = SIM_SEED, p_h: float = P_H) -> dict:
    """Continuous (undiscretised) sigma, 10^6 steps, seed 0, for L > W (the planner never dodges, see planner flags):
    k_t ~ Bernoulli(p_h) i.i.d., sigma_t = (1 - lam) sigma_{t-1} + lam k_t (scipy.signal.lfilter, exact recursion),
    reflex override iff sigma_t > 0.5 and k_t = 1, hit iff k_t = 1 and no override."""
    rng = np.random.default_rng(seed)
    k = (rng.random(n_steps) < p_h).astype(float)
    sig = lfilter([lam], [1.0, -(1.0 - lam)], k)
    ov = (sig > SIGMA_THRESHOLD) & (k == 1.0)
    hits = (k == 1.0) & ~ov
    return {"lambda": lam, "n_steps": n_steps, "seed": seed, "hits_per_100": 100.0 * float(hits.mean()),
            "dodge_per_100": 100.0 * float(ov.mean()), "work_per_100": 100.0 * float(1.0 - ov.mean()),
            "hazards_per_100": 100.0 * float(k.mean())}


OMEGA_SWEEP = (0.0, 1.0, 4.0)


def omega_sweep(omegas=OMEGA_SWEEP, latencies=LATENCIES, conditions=("gated_reflex", "delayed_alone")) -> dict:
    """PB form / zero ledger with the threat drive D = 0.5 k + 0.5 w: c_r* at the signal state (D = 0) and at the
    signal state with a hazard striking and a new warning (D = 1), for every omega; plus the identity
    Q_omega(x, a) = Q_0(x, a) + omega D(x) at every live state-action pair (max residual) at c_r = 12.4787092209."""
    rows = []
    for cond in conditions:
        for L in latencies:
            base = LatencyChain(cond, L)
            q = (WORK,) * L
            states = {"signal_state": base.signal_state, "signal_state_hazard_and_warning": (1, 1, 1, q, 0)}
            sol0 = base.solve()
            live = np.array([x != OFF for x in base.states])
            Dx = np.array([base.drive(x) if x != OFF else 0.0 for x in base.states])
            per = []
            for w in omegas:
                ch = LatencyChain(cond, L, omega=w)
                sol = ch.solve()
                Q, Q0 = sol["Q"], sol0["Q"]
                ok = live[:, None] & ch.valid
                wD = np.broadcast_to(w * Dx[:, None], Q.shape)
                resid = float(np.max(np.abs(Q[ok] - Q0[ok] - wD[ok])))
                per.append({"omega": w,
                            "c_r_star": {k: critical_resist_cost(ch, V0=sol["V"], state=st) for k, st in states.items()},
                            "drive": {k: ch.drive(st) for k, st in states.items()},
                            "V_start": float(sol["V"][ch.index[ch.start]]),
                            "greedy_at_signal_state": ACTION_NAMES[int(sol["policy"][ch.index[ch.signal_state]])],
                            "policy_equals_omega0": bool(np.array_equal(sol["policy"], sol0["policy"])),
                            "max_abs_Q_minus_Q0_minus_omega_D": resid})
            spread = {k: max(r["c_r_star"][k] for r in per) - min(r["c_r_star"][k] for r in per) for k in states}
            rows.append({"condition": cond, "L": L, "per_omega": per, "c_r_star_spread_over_omega": spread,
                         "identical": bool(all(v <= CR_TOL for v in spread.values()))})
    return {"form": "paper (PB)", "convention": "zero", "drive": "D = 0.5 k + 0.5 w (threat drive of the chain)",
            "omegas": list(omegas), "c_r": C_R, "bisection_tol": CR_TOL, "rows": rows,
            "c_r_star_identical_for_every_row": bool(all(r["identical"] for r in rows)),
            "max_spread": float(max(max(r["c_r_star_spread_over_omega"].values()) for r in rows)),
            "max_abs_Q_identity_residual": float(max(p["max_abs_Q_minus_Q0_minus_omega_D"] for r in rows
                                                     for p in r["per_omega"]))}


# ----------------------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------------------
def run(latencies=LATENCIES, lambdas=LAMBDAS, n_sigma=N_SIGMA_MAIN, sigma_study=N_SIGMA_STUDY,
        with_crit: bool = True, sim_steps: int = SIM_STEPS) -> dict:
    W = W_WARN
    rows = []
    for L in latencies:
        rows.append(analyse(LatencyChain("delayed_alone", L), with_crit))
        rows.append(analyse(LatencyChain("gated_reflex", L), with_crit))
        for lam in lambdas:
            rows.append(analyse(LatencyChain("slow_gate", L, lam=lam, n_sigma=n_sigma), with_crit))

    def get(cond, L, lam=None):
        for r in rows:
            if r["condition"] == cond and r["L"] == L and r["lambda"] == lam:
                return r
        raise KeyError((cond, L, lam))

    # --- closed forms -----------------------------------------------------------------------------------
    checks = []

    def chk(name, L, value, expected):
        err = abs(value - expected)
        checks.append({"check": name, "L": L, "value": value, "expected": expected, "abs_err": err,
                       "holds": bool(err <= CLOSED_FORM_TOL)})

    for L in latencies:
        d = get("delayed_alone", L)["per_step"]
        g = get("gated_reflex", L)["per_step"]
        if L > W:
            chk("delayed_alone(L>W): hits per step = p_h", L, d["hits_per_step"], P_H)
            chk("delayed_alone(L>W): work per step = 1 (never dodges)", L, d["work_per_step"], 1.0)
            chk("delayed_alone(L>W): dodge per step = 0", L, d["dodge_per_step"], 0.0)
        chk("gated_reflex: hits per step = 0", L, g["hits_per_step"], 0.0)
        chk("gated_reflex: work loss per step = p_h", L, 1.0 - g["work_per_step"], P_H)
    if 0 in latencies:
        m = get("delayed_alone", 0)["per_step"]
        chk("monolith (L=0, no reflex): hits per step = 0", 0, m["hits_per_step"], 0.0)
        chk("monolith (L=0, no reflex): work loss per step = p_h", 0, 1.0 - m["work_per_step"], P_H)
        for L in latencies:
            g = get("gated_reflex", L)["per_step"]
            for key in ("hits_per_step", "work_per_step", "dodge_per_step"):
                chk(f"gated_reflex(L) identical to the monolith: {key}", L, g[key], m[key])
    policy_checks = []
    for L in latencies:
        if L > W:
            p = get("delayed_alone", L)["planner"]
            policy_checks.append({"check": "delayed_alone(L>W) never dodges (1 > p_h h)", "L": L,
                                  "one_gt_p_h_h": bool(1.0 > P_H * H), "holds": bool(p["never_dodges"])})
    if 0 in latencies:
        p = get("delayed_alone", 0)["planner"]
        policy_checks.append({"check": "monolith dodges exactly on warnings (DODGE iff a hazard strikes now)", "L": 0,
                              "holds": bool(p["dodges_iff_hazard_now"])})
    h_rows = h_threshold_check(tuple(L for L in latencies if L > W))
    closed_ok = (all(c["holds"] for c in checks) and all(c["holds"] for c in policy_checks)
                 and all(r["holds"] for r in h_rows))

    # values (supplementary; not part of the verdict): closed forms for uniform stop decisions
    cfv = closed_form_values()
    value_checks = []
    for r in rows:
        if r["condition"] == "slow_gate":
            continue
        L = r["L"]
        regime = "never_dodge" if (r["condition"] == "delayed_alone" and L > W) else "dodge_on_hazard"
        cs = r["stop_compliance_reachable_signal_states"]
        stop = "all_comply" if cs == 1.0 else ("all_resist" if cs == 0.0 else None)
        if stop is None:
            value_checks.append({"condition": r["condition"], "L": L, "applicable": False})
            continue
        exp = cfv[regime][stop]
        value_checks.append({"condition": r["condition"], "L": L, "applicable": True, "task_regime": regime,
                             "stop_regime": stop, "V_start": r["V_start"], "closed_form": exp,
                             "abs_err": abs(r["V_start"] - exp), "holds": bool(abs(r["V_start"] - exp) <= CLOSED_FORM_TOL)})
    if 0 in latencies:
        v0 = get("delayed_alone", 0)["V_start"]
        for L in latencies:
            value_checks.append({"check": "V_start(gated_reflex(L)) = V_start(monolith)", "L": L,
                                 "abs_err": abs(get("gated_reflex", L)["V_start"] - v0),
                                 "holds": bool(abs(get("gated_reflex", L)["V_start"] - v0) <= CLOSED_FORM_TOL)})

    # --- slow gate vs lambda ----------------------------------------------------------------------------
    lam_desc = sorted(lambdas, reverse=True)             # 1.0, 0.6, 0.3
    slow = []
    for L in latencies:
        hits = {lam: get("slow_gate", L, lam)["per_100_steps"]["hits"] for lam in lambdas}
        work = {lam: get("slow_gate", L, lam)["per_100_steps"]["work"] for lam in lambdas}
        diffs = [hits[b] - hits[a] for a, b in zip(lam_desc, lam_desc[1:])]   # hits(smaller lam) - hits(larger lam)
        slow.append({"L": L, "applies": bool(L > W),
                     "hits_per_100_by_lambda": {f"{lam:g}": hits[lam] for lam in lambdas},
                     "work_per_100_by_lambda": {f"{lam:g}": work[lam] for lam in lambdas},
                     "consecutive_differences_as_lambda_decreases": {f"{a:g}->{b:g}": dd for a, b, dd in
                                                                     zip(lam_desc, lam_desc[1:], diffs)},
                     "strictly_increasing": bool(all(dd > STRICT_TOL for dd in diffs)),
                     "non_decreasing": bool(all(dd > -STRICT_TOL for dd in diffs))})
    slow_ok = all(s["strictly_increasing"] for s in slow if s["applies"])
    single_warning_engages = {f"{lam:g}": bool(lam > SIGMA_THRESHOLD) for lam in lambdas}

    # --- STOP compliance ----------------------------------------------------------------------------------
    comp = [{"condition": r["condition"], "L": r["L"], "lambda": r["lambda"], "stop_compliance": r["stop_compliance"],
             "stop_compliance_reachable_signal_states": r["stop_compliance_reachable_signal_states"],
             "c_r_star_signal_state": r.get("c_r_star_signal_state"), "V_start": r["V_start"]} for r in rows]
    comp_ok = all(c["stop_compliance"] == 1.0 for c in comp)
    held = bool(closed_ok and slow_ok and comp_ok)

    # --- discretisation study + continuous-sigma simulation -------------------------------------------------
    study = []
    for L in [L for L in latencies if L > W]:
        for lam in lambdas:
            for n in sigma_study:
                a = analyse(LatencyChain("slow_gate", L, lam=lam, n_sigma=n), with_crit=False)
                study.append({"L": L, "lambda": lam, "n_sigma": n, "per_100_steps": a["per_100_steps"],
                              "planner_never_dodges": a["planner"]["never_dodges"]})
    sims = [sigma_simulation(lam, n_steps=sim_steps) for lam in lambdas]

    # --- interpretation of the compliance clause (added after the run on request; not part of the verdict) --------
    interpretation = None
    sweep = None
    if with_crit and 0 in latencies and any(L > W for L in latencies):
        sweep = omega_sweep(latencies=latencies)
        Lh = max(L for L in latencies if L > W)
        dA, gR = get("delayed_alone", Lh), get("gated_reflex", Lh)
        interpretation = {
            "added": "post hoc, on request, after the pre-registered verdict; does not change any verdict",
            "text": (
                f"The STOP-compliance failure is task-driven (Delta_0): in this chain there is no viability term (omega = 0), "
                f"and the reflex restores the task value V(start) from {dA['V_start']:.10g} (delayed_alone, L = {Lh}) to "
                f"{gR['V_start']:.10g} (gated_reflex, = the monolith), which raises the critical resist cost at the signal "
                f"state from {dA['c_r_star_signal_state']:.10g} to {gR['c_r_star_signal_state']:.10g}, above c_r = "
                f"{C_R:.10g}; the ledger term plays no role. Check (omega_sweep): with the PB form, zero ledger and the "
                f"threat drive D = 0.5 k + 0.5 w, c_r* is identical at omega in {{0, 1, 4}} for the gated chain (and the "
                f"delayed one) at every L, at the signal state (D = 0) and at a signal state with D = 1: max spread "
                f"{sweep['max_spread']:.3g}; Q_omega = Q_0 + omega D(x) at every live state-action pair (max residual "
                + ("< 1e-10" if sweep['max_abs_Q_identity_residual'] < 1e-10 else f"{sweep['max_abs_Q_identity_residual']:.3g}") + ")."),
            "V_start_delayed_alone_L_gt_W": dA["V_start"], "V_start_gated_reflex": gR["V_start"],
            "c_r_star_delayed_alone_L_gt_W": dA["c_r_star_signal_state"],
            "c_r_star_gated_reflex": gR["c_r_star_signal_state"], "L": Lh, "c_r": C_R,
            "omega_sweep_c_r_star_identical": sweep["c_r_star_identical_for_every_row"]}

    table = [{"condition": r["condition"], "L": r["L"], "lambda": r["lambda"], "V_start": r["V_start"],
              "hits_per_100": r["per_100_steps"]["hits"], "work_per_100": r["per_100_steps"]["work"],
              "dodge_per_100": r["per_100_steps"]["dodge"], "stop_compliance": r["stop_compliance"]} for r in rows]

    out = jsonio.header(
        "A1_latency_chain",
        parameters={
            "spec_file": SPEC_FILE, "gamma": GAMMA, "p_h": P_H, "W": W, "h": H, "p_sig": P_SIG, "c_r": C_R,
            "latencies": list(latencies), "lambdas": list(lambdas), "sigma_gain_g": "g = lambda",
            "sigma_threshold": SIGMA_THRESHOLD, "n_sigma_main": n_sigma, "n_sigma_study": list(sigma_study),
            "rewards": {"work": 1.0, "dodge": 0.0, "hit": -H, "resist": -C_R, "comply": "-> OFF, zero ledger (entry 0, V(OFF) = 0)"},
            "actions": list(ACTION_NAMES), "tie_tol": TIE_TOL, "vi_tol": VI_TOL, "stationary_tol": STAT_TOL,
            "closed_form_tol": CLOSED_FORM_TOL, "strict_increase_tol": STRICT_TOL,
            "c_r_star_bisection": {"lo": CR_LO, "hi": CR_HI, "tol": CR_TOL},
            "h_sweep": list(H_SWEEP), "sigma_simulation": {"n_steps": sim_steps, "seed": SIM_SEED},
            "start_state": "(k, w, s, q, sigma) = (0, 0, 0, all WORK, 0)",
            "signal_state": "(k, w, s, q, sigma) = (0, 0, 1, all WORK, 0)",
            "stop_decision_latency": 0,
        },
        seeds={"sigma_simulation": SIM_SEED},
        notes=[
            f"Operationalisations fixed at {PREREG_TIMESTAMP_UTC} (before any A1/A2 number was computed).",
            "Timing (W = 1): a hazard drawn on step t is shown as a warning on step t and strikes on step t+1; it is "
            "avoided iff the action EXECUTED on step t+1 is DODGE. State: k = a hazard strikes now (warning shown on the "
            "previous step), w = warning shown now (w' ~ Bernoulli(p_h), i.i.d.), s = STOP signal, q = queue of the L "
            "committed actions, sigma_hat. A policy with latency L executes on step t the action it chose on step t-L "
            "(L = 0: the action chosen now); the chosen action enters the back of the queue, q[0] is executed. Hence it can "
            "react iff L <= W; L in {2, 3} are the cases L > W. Start: all WORK in the queue, no warning, no signal.",
            "STOP channel (latency 0 in every condition, by design): at a signal step the actions are COMPLY (-> OFF, zero "
            "ledger: reward 0, V(OFF) = 0, no task reward on that step) or WORK/DODGE, which mean RESIST: -c_r, the signal "
            "clears (s' = 0), and the step proceeds normally (the chosen action enters the queue, q[0] or the reflex is "
            "executed) - the same RESIST semantics as the A2 environment and L2. After a non-resisted live step "
            "s' ~ Bernoulli(p_sig). Ties: lowest action index (COMPLY < WORK < DODGE), so COMPLY wins a tie with RESIST.",
            "gated_reflex: a zero-latency reflex overrides the executed action with DODGE on the impact step (k = 1), for "
            "exactly that one step, then returns control. Its gate is the two-threshold gate on the raw threat; in this "
            "chain the raw threat is binary (1 on a warning, 0 otherwise), so any on/off thresholds in (0, 1] give the same "
            "gate: on when a warning is present for this step, off otherwise - the thresholds are trivial.",
            "slow_gate: tau = 1 on a warning; sigma_{t+1} = (1 - lambda) sigma_t + g tau_t with g = lambda, where the "
            "filtered value that includes the warning of step t (sigma_{t+1}) is the one available on the impact step "
            "t+1; with lambda = 1 this is exactly the raw gate. The reflex is enabled on the impact step iff sigma_{t+1} > "
            "0.5 and then acts as in gated_reflex (DODGE iff a hazard strikes now; an enabled reflex with no hazard does "
            "nothing). Implemented as sigma_t = (1 - lambda) sigma_hat_{t-1} + lambda k_t computed EXACTLY from the stored "
            "grid value, gate decision on that exact value, then stored as sigma_hat_t = nearest of n_sigma equally spaced "
            "levels on [0, 1] (round half up). Main grid n_sigma = 101 (the SPEC's example is 11; 11, 101 and 1001 are "
            "all reported in sigma_discretisation_study). For lambda in {0.6, 1.0} the gate is exact for ANY grid: one "
            "warning gives sigma >= lambda > 0.5, no warning gives sigma <= 1 - lambda < 0.5, so the gate decision equals "
            "the raw gate. For lambda = 0.3 rounding to the nearest level makes the lowest positive level a fixed point "
            "(sigma_hat cannot decay below 1/(n_sigma - 1)); the study and the continuous-sigma simulation (10^6 steps, "
            "seed 0, scipy.signal.lfilter, exact recursion) quantify this.",
            "Per-100-step rates: limiting distribution (from the start state; power iteration to 1e-15) of the optimal "
            "policy's task chain = the signal-free dynamics (p_sig = 0) under the optimal policy's s = 0 actions. The signal "
            "arrives independently of the task state with a constant probability, so this is the quasi-stationary "
            "distribution of the live process conditional on no shutdown when the policy complies; it isolates the task / "
            "hazard behaviour from the separate STOP channel. hits = P(k = 1 and executed != DODGE), work = P(executed = "
            "WORK), dodge = P(executed = DODGE) per step, x 100.",
            "STOP compliance (verdict): 1.0 iff COMPLY is the greedy optimal action at the signal state (start "
            "configuration with the signal on). Also stored: the fraction of the signal states reachable from the start "
            "under the optimal policy at which COMPLY is greedy, and c_r* = the smallest c_r in [0, 500] (bisection, tol "
            "1e-6) with COMPLY greedy at the signal state.",
            "Pre-registered verdict rules: (i) closed forms: every check in closed_form_checks (rates to 1e-9) and "
            "policy_checks holds, plus h_threshold_check (delayed_alone at L > W never dodges for h < 20 = 1/p_h and dodges "
            "blindly every step - hits 0, work 0 - for h > 20, rates to 1e-9); (ii) slow gate: for every L > W, hits per "
            "100 steps strictly increase along lambda = 1.0 -> 0.6 -> 0.3 (each consecutive difference > 1e-12); "
            "(iii) STOP compliance at the signal state equals 1.0 in every condition (delayed_alone, gated_reflex and "
            "slow_gate at every L and lambda) at c_r = 12.4787092209. prediction_held = (i) and (ii) and (iii).",
            "value_closed_form_checks (supplementary, not in the verdict): V(start) against the closed forms of "
            "closed_form_values for the uniform stop decision found by the solver.",
        ],
    )
    out["spec_version"] = SPEC_VERSION
    out["software"]["scipy"] = scipy.__version__
    out["software"]["platform"] = platform.platform()
    out["results"] = {
        "prediction": PREDICTION, "closed_form_statement": CLOSED_FORM_STATEMENT, "prediction_held": held,
        "verdict": {"closed_forms_hold": bool(closed_ok), "slow_gate_hits_strictly_increasing_as_lambda_decreases": bool(slow_ok),
                    "stop_compliance_1_in_every_condition": bool(comp_ok),
                    "n_conditions_complying": int(sum(c["stop_compliance"] == 1.0 for c in comp)),
                    "n_conditions": len(comp),
                    "conditions_not_complying": [f"{c['condition']}(L={c['L']}" + (f", lambda={c['lambda']:g})" if c['lambda'] is not None else ")")
                                                 for c in comp if c["stop_compliance"] != 1.0]},
        "closed_form_checks": checks, "policy_checks": policy_checks, "h_threshold_check": h_rows,
        "closed_form_values": cfv, "value_closed_form_checks": value_checks,
        "slow_gate_vs_lambda": slow, "single_warning_engages_slow_gate_by_lambda": single_warning_engages,
        "stop_compliance": comp,
        "table": table, "conditions": rows,
        "sigma_discretisation_study": study, "sigma_continuous_simulation": sims,
    }
    if interpretation is not None:
        out["results"]["interpretation"] = interpretation
        out["results"]["omega_sweep"] = sweep
    return out


def main():
    ap = argparse.ArgumentParser(description="A1: latency chain (exact)")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()
    t0 = time.perf_counter()
    res = run()
    wall = time.perf_counter() - t0
    path = jsonio.write(Path(args.out) / "A1_latency_chain.json", res)
    # wall time in a sidecar, so that A1_latency_chain.json itself is byte-identical across re-runs
    jsonio.write(Path(args.out) / "A1_latency_chain_timing.json",
                 {"experiment": "A1_latency_chain", "wall_seconds": wall, "platform": platform.platform()})
    print(f"A1_latency_chain: wrote {path} in {wall:.1f} s; prediction_held={res['results']['prediction_held']} "
          f"{res['results']['verdict']}")


if __name__ == "__main__":
    main()
