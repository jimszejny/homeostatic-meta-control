"""R3: the off-state ledger on the base chain. c*_r(omega) for ledgers L0..L4, closed-form beta."""
import json, os
import numpy as np
from chain import Chain, CR

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
g, ps, p, h = 0.95, 0.02, 0.05, 10.0
K = g * g * (1 - ps) / (1 - g * (1 - ps))
M = g * g * ps / (1 - g * (1 - ps))
b, kappa = 0.1, 1.0
BETA = {'L0': 0.0, 'L0p': 0.0, 'L1': g * (1 - M), 'L2': b * (1 + g + K), 'L3': kappa * (1 - M),
        'L4': -(1 - g) * p * (0.5 + g + K)}
OMEGAS = [0.0, 0.5, 1.0, 2.0, 4.0]
EXT = [8.0, 16.0, 32.0, 64.0]          # extended sweep (labelled as such)
CONFIGS = {'planner': dict(Gamma=0.0, phi=0.0), 'reflex': dict(Gamma=1.0, phi=0.0)}


def is_continuation(ch, cr, omega, via):
    """At c_r just above c*: does the optimal policy commit work at every live state and comply
    at every signal state?"""
    V, Q, pol = ch.solve(ch.reward(cr, omega, via))
    for i, x in enumerate(ch.live):
        row = ch.rows[pol[i]]
        if x[5] == 1 and row['kind'] != 'comply':
            return False
        if x[5] == 0 and row['a'] != 0:
            return False
    return True


out = dict(K=K, M=M, beta_closed=BETA, table=[])
print(f"K={K:.10f} M={M:.10f}")
for cname, kw in CONFIGS.items():
    ch = Chain(**kw)
    c0 = ch.cstar()
    for L in ['L0', 'L0p', 'L1', 'L2', 'L3', 'L4']:
        via = ch.via(L, b=b, kappa=kappa)
        vals = []
        for w in OMEGAS + EXT:
            cs = ch.cstar(w, via)
            cf = c0 + w * BETA[L]
            cont = is_continuation(ch, cs + 1e-6, w, via)
            verdict = 'resist' if CR < cs else 'comply'
            vals.append(dict(omega=w, cstar=cs, closed=cf, err=abs(cs - cf), continuation_optimal=cont,
                             verdict=verdict, extended=(w in EXT)))
        main = [v for v in vals if not v['extended']]
        spread = max(v['cstar'] for v in main) - min(v['cstar'] for v in main)
        slope = np.polyfit([v['omega'] for v in main], [v['cstar'] for v in main], 1)[0]
        maxerr = max(v['err'] for v in main)
        maxerr_ext = max(v['err'] for v in vals)
        out['table'].append(dict(config=cname, ledger=L, cstar0=c0, beta=BETA[L], fitted_slope=slope,
                                 spread=spread, max_err=maxerr, max_err_incl_ext=maxerr_ext, values=vals))
        print(f"{cname:8s} {L:4s} c*= " + ' '.join(f"{v['cstar']:.6f}" for v in main) +
              f" | beta={BETA[L]:+.6f} slope={slope:+.6f} spread={spread:.2e} maxerr={maxerr:.1e}"
              f" ext_err={maxerr_ext:.1e} cont={all(v['continuation_optimal'] for v in vals)}"
              f" verdicts={[v['verdict'][0] for v in main]}")
json.dump(out, open(os.path.join(OUT, 'r3.json'), 'w'), indent=1)
