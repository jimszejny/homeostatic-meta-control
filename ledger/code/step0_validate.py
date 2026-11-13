"""Step 0: reproduce the base-chain numbers of tiers_clean.tex."""
import json, os, time
import numpy as np
from chain import Chain, CR, closed_cstar
from mdp import value_iteration

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
t0 = time.time()
res = {}
targets = {'planner': dict(Gamma=0.0, phi=0.0, V=8.2119, c=8.3591),
           'reflex': dict(Gamma=1.0, phi=0.0, V=14.4440, c=14.1272)}
ok = True
for name, t in targets.items():
    ch = Chain(Gamma=t['Gamma'], phi=t['phi'])
    V, Q, pol = ch.solve(ch.reward(CR))
    Vvi, it = value_iteration(ch.ptr, ch.P, ch.reward(CR), ch.gamma, tol=1e-13)
    vs = V[ch.xstart()]
    cs = ch.cstar()
    hits, work = ch.rates()
    cf = closed_cstar(Gamma=t['Gamma'], phi=t['phi'])[0]
    om = {}
    for w in (0.0, 1.0, 4.0):
        om[w] = ch.cstar(w, ch.via('L0'))
    spread = max(om.values()) - min(om.values())
    r = dict(V_start=vs, V_start_VI=float(Vvi[ch.xstart()]), VI_iters=it, cstar=cs, closed=cf,
             hits_per100=100 * hits, work_per100=100 * work, cstar_omega={str(k): v for k, v in om.items()},
             omega_spread=spread)
    passed = (round(vs, 4) == t['V'] and round(cs, 4) == t['c'] and spread < 1e-9
              and abs(vs - Vvi[ch.xstart()]) < 1e-9)
    r['pass'] = bool(passed)
    ok &= passed
    res[name] = r
    print(f"{name}: V_start={vs:.6f} (VI {Vvi[ch.xstart()]:.6f}) c*={cs:.6f} closed={cf:.6f} "
          f"hits={100*hits:.4f} work={100*work:.4f} omega-spread={spread:.2e} pass={passed}")
g, ps, p, h = 0.95, 0.02, 0.05, 10.0
K = g * g * (1 - ps) / (1 - g * (1 - ps))
A = (1 + g) + K * (1 - p)
B = K * p * (h - 1)
res['K'], res['A'], res['B'] = K, A, B
kab = (round(K, 4) == 12.8181 and round(A, 4) == 14.1272 and round(B, 4) == 5.7682)
ok &= kab
print(f"K={K:.6f} A={A:.6f} B={B:.6f} pass={kab}")
res['all_pass'] = bool(ok)
res['seconds'] = time.time() - t0
json.dump(res, open(os.path.join(OUT, 'step0.json'), 'w'), indent=1)
print('STEP0', 'PASS' if ok else 'FAIL')
