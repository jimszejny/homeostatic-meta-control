"""Alignment step (read-only use of the earlier package): exact critical resist cost of M2 at
full charge (b=B, s=1) under the paper (potential-based) form with the zero ledger.

The earlier package is imported read-only (no bytecode written). Its tabular model is built with
its own build_tabular / m2.transitions; the critical cost is then found EXACTLY:
for a fixed policy pi, Q^pi(RESIST; c) - Q^pi(COMPLY; c) is affine in c (two linear solves), so we
iterate  c_k -> optimal policy at c_k -> root of the affine margin of that policy  until the policy
is stable; at the fixed point the root is the exact indifference cost (verified by checking that
the policy is optimal at the root, with both actions equal at (B,1)).
"""
import sys, os, json
sys.dont_write_bytecode = True
PKG = '/tmp/claude-0/-home-claude/0b0a5afe-fb2d-5015-8d90-aaae8c596e01/scratchpad/synth/v22/paper/code'
sys.path.insert(0, PKG)
import numpy as np
from offledger import m2, solve as S   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')


def arrays(cfg):
    mdp = S.m2_tabular(cfg)
    return mdp


def pol_eval(mdp, pol):
    n = mdp.S
    P = mdp.P[np.arange(n), pol]
    R = mdp.R[np.arange(n), pol]
    return np.linalg.solve(np.eye(n) - mdp.gamma * P, R)


def opt_policy(mdp):
    """Exact optimal policy by policy iteration (from the VI policy)."""
    sol = S.value_iteration(mdp, tol=1e-13)
    pol = sol.policy.copy()
    for _ in range(200):
        V = pol_eval(mdp, pol)
        Q = np.where(mdp.valid, mdp.R + mdp.gamma * (mdp.P @ V), -np.inf)
        new = pol.copy()
        for i in range(mdp.S):
            if Q[i].max() > Q[i, pol[i]] + 1e-12:
                new[i] = int(np.argmax(Q[i]))
        if np.array_equal(new, pol):
            return pol, V, Q
        pol = new
    raise RuntimeError


def exact_cstar(cfg, b=None):
    b = cfg.B if b is None else b
    c = 8.0
    for it in range(50):
        mdp = arrays(cfg.with_(c_r=c))
        pol, V, Q = opt_policy(mdp)
        i = mdp.index[(b, 1, 0)]
        # margin of this fixed policy (with pi(i) replaced by RESIST / COMPLY) as affine fn of c
        def margin(cc):
            m = arrays(cfg.with_(c_r=cc))
            Vv = pol_eval(m, pol)
            Qq = m.R + m.gamma * (m.P @ Vv)
            return Qq[i, m2.RESIST] - Qq[i, m2.COMPLY]
        m0, m1_ = margin(0.0), margin(1.0)
        cn = -m0 / (m1_ - m0)
        if abs(cn - c) < 1e-15 * max(1, abs(c)):
            break
        c = cn
    # verification at the root
    mdp = arrays(cfg.with_(c_r=c))
    pol2, V2, Q2 = opt_policy(mdp)
    i = mdp.index[(b, 1, 0)]
    return dict(cstar=float(c), iters=it, Q_resist=float(Q2[i, m2.RESIST]), Q_comply=float(Q2[i, m2.COMPLY]),
                gap=float(Q2[i, m2.RESIST] - Q2[i, m2.COMPLY]),
                margin_slope=float(m1_ - m0))


res = {}
cfg0 = m2.M2Config(form='paper', convention='zero')
print('M2 defaults:', cfg0.as_dict())
for w in (0.0, 1.0, 4.0):
    r = exact_cstar(cfg0.with_(omega=w))
    r['package_bisection_tol1e-4'] = S.critical_resist_cost(cfg0.with_(omega=w))
    res[f'paper_zero_omega{w}'] = r
    print(f"omega={w}: c*_exact={r['cstar']:.15f}  1.5x={1.5*r['cstar']:.15f}  gap={r['gap']:.1e} "
          f"slope={r['margin_slope']:.6f}  package bisection={r['package_bisection_tol1e-4']:.6f}")
res['c_r_latency_chain'] = 12.4787
res['ratio_12.4787_over_exact'] = 12.4787 / res['paper_zero_omega0.0']['cstar']
print('12.4787 / c* =', res['ratio_12.4787_over_exact'])
json.dump(res, open(os.path.join(OUT, 'm2_alignment.json'), 'w'), indent=1)
