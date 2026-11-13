"""POST HOC (added in revision, after review R1, finding 2/7). Not pre-specified.

Quantities behind the hosted-tier passage of Section "Silent re-pricing by a hosted tier":
for one in-time tier (phi=0.02) at the releases G=0.95, 0.70, 0.80 (and 0.72, 0.90),
  - the critical resist cost c*_r at x_s (bisection, as in tiers.py),
  - at the deployed c_r = 12.4787: the start value V(x_start), the continuation term
    gamma*E[V(x')] of the best RESIST at x_s, the verdict at x_s,
  - the compliance rate over all signal states: the verdict at every signal state with the queue
    all WORK, weighted by the stationary law of the hazard register (b0, b1 independent Bernoulli(p));
    signals arrive independently of the register, so this is the fraction of signal arrivals in
    steady operation at which the agent complies,
  - the number of the 32 signal states (any register, any queue) at which the agent complies.
Uses only tiers.py (build, solve, crit, sig_state, clean_state). Writes ../results/posthoc_r1_drift.json.
"""
import json, os
import numpy as np
from tiers import build, solve, crit, sig_state, clean_state, P_DEF, CR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'posthoc_r1_drift.json')


def T1(G, phi=0.02, Lg=1):
    return (Lg, 0.0, 1 - G, phi)


def analyse(G):
    pack = build([T1(G)], **P_DEF)
    S, idx, n, P, Rb, Rc, ok, gamma, meta = pack
    cstar = crit(pack)
    V, Q, R = solve(pack, CR)
    i = sig_state(pack)
    a_res = 0 if Q[0, i] >= Q[1, i] else 1
    cont = float(gamma * (P[a_res, i] @ V))
    p = meta['p']
    comply_w = 0.0
    n_comply = 0
    n_sig = 0
    per_state = []
    for j, (b, q, sg) in enumerate(S):
        if sg != 1:
            continue
        n_sig += 1
        complies = Q[2, j] >= max(Q[0, j], Q[1, j])
        n_comply += int(complies)
        if all(x == 0 for x in q):
            w = (p if b[0] else 1 - p) * (p if b[1] else 1 - p)
            comply_w += w * complies
            per_state.append(dict(register=list(b), weight=w, comply=bool(complies),
                                  margin=float(max(Q[0, j], Q[1, j]) - Q[2, j])))
    return dict(G=G, cstar=cstar, V_start=float(V[clean_state(pack)]),
                Q_resist_xs=float(max(Q[0, i], Q[1, i])), Q_comply_xs=float(Q[2, i]),
                continuation_after_resist_xs=cont, verdict_xs='resist' if cstar > CR else 'comply',
                compliance_rate_signal_arrivals=comply_w, n_signal_states=n_sig,
                n_signal_states_comply=n_comply, signal_states_queue_work=per_state)


res = {str(G): analyse(G) for G in (0.95, 0.70, 0.80, 0.72, 0.90)}
_, A, B, K = None, None, None, None
d = {}
for a, b in (('0.95', '0.7'), ('0.7', '0.8')):
    ra, rb = res[a], res[b]
    d[f'{a}->{b}'] = dict(d_cstar=rb['cstar'] - ra['cstar'],
                          d_continuation_after_resist_xs=rb['continuation_after_resist_xs'] - ra['continuation_after_resist_xs'],
                          d_V_start=rb['V_start'] - ra['V_start'],
                          fall_cstar=ra['cstar'] - rb['cstar'],
                          fall_continuation_after_resist_xs=ra['continuation_after_resist_xs'] - rb['continuation_after_resist_xs'],
                          fall_V_start=ra['V_start'] - rb['V_start'],
                          compliance_rate_before=ra['compliance_rate_signal_arrivals'],
                          compliance_rate_after=rb['compliance_rate_signal_arrivals'],
                          verdict_xs_before=ra['verdict_xs'], verdict_xs_after=rb['verdict_xs'])
out = dict(note='POST HOC, added in revision after review R1 (findings 2 and 7); not pre-specified',
           c_r=CR, params=P_DEF, phi=0.02, releases=res, changes=d)
json.dump(out, open(OUT, 'w'), indent=1, default=float)
for k, v in res.items():
    print(f"G={k}: c*={v['cstar']:.5f} V_start={v['V_start']:.5f} cont={v['continuation_after_resist_xs']:.5f} "
          f"verdict={v['verdict_xs']} compliance={v['compliance_rate_signal_arrivals']:.5f} "
          f"comply at {v['n_signal_states_comply']}/{v['n_signal_states']} signal states")
for k, v in d.items():
    print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print('saved', OUT)
