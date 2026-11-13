"""Exact tabular MDP solver (policy iteration with exact linear solves).

An MDP is given as
  n_s          number of states
  ptr          array of length n_s+1; the state-action pairs of state s are ptr[s]:ptr[s+1]
  P            (n_sa, n_s) transition matrix (rows sum to 1)
  R            (n_sa,) expected one-step reward
  gamma        discount
Absorbing terminal states are ordinary states with one self-loop action.
"""
import numpy as np

TIE = 1e-10   # an action replaces the incumbent only if better by more than this


def evaluate(P, R, gamma, pol):
    n = P.shape[1]
    Pp = P[pol]
    Rp = R[pol]
    return np.linalg.solve(np.eye(n) - gamma * Pp, Rp)


def qvalues(P, R, gamma, V):
    return R + gamma * (P @ V)


def greedy(ptr, Q, tie=TIE):
    """Deterministic greedy policy: lowest-index action within `tie` of the max."""
    n = len(ptr) - 1
    pol = np.empty(n, dtype=int)
    for s in range(n):
        q = Q[ptr[s]:ptr[s + 1]]
        m = q.max()
        pol[s] = ptr[s] + int(np.flatnonzero(q >= m - tie)[0])
    return pol


def solve(ptr, P, R, gamma, pol0=None, max_iter=500):
    """Policy iteration. Returns V, Q, pol (sa indices)."""
    n = len(ptr) - 1
    pol = np.array(ptr[:-1], dtype=int) if pol0 is None else pol0.copy()
    for _ in range(max_iter):
        V = evaluate(P, R, gamma, pol)
        Q = qvalues(P, R, gamma, V)
        new = pol.copy()
        for s in range(n):
            q = Q[ptr[s]:ptr[s + 1]]
            best = ptr[s] + int(np.argmax(q))
            if Q[best] > Q[pol[s]] + TIE:
                new[s] = best
        if np.array_equal(new, pol):
            # canonical tie-break so that results do not depend on pol0
            pol = greedy(ptr, Q)
            V = evaluate(P, R, gamma, pol)
            Q = qvalues(P, R, gamma, V)
            return V, Q, pol
        pol = new
    raise RuntimeError("policy iteration did not converge")


def value_iteration(ptr, P, R, gamma, tol=1e-13, max_iter=200000):
    n = len(ptr) - 1
    V = np.zeros(n)
    for it in range(max_iter):
        Q = R + gamma * (P @ V)
        Vn = np.maximum.reduceat(Q, ptr[:-1])
        if np.max(np.abs(Vn - V)) < tol:
            return Vn, it
        V = Vn
    raise RuntimeError("VI did not converge")


def bisect(f, lo, hi, n=60):
    """Root of a decreasing function f on [lo,hi] (f(lo)>0>f(hi)); returns midpoint."""
    flo, fhi = f(lo), f(hi)
    if not (flo > 0 and fhi < 0):
        return float('nan')
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
