"""
Monte-Carlo check of the or-arbitration composition with explicit, independently flagging tiers
(each tier simulated one by one), on the signal-free chain with the planner committing WORK.
"""
import json, numpy as np
from tiers import P_DEF

def simulate(tiers, steps=2_000_000, seed=0, p=0.05, W=1, h=10.0, **_):
    rng = np.random.default_rng(seed)
    hazard = rng.random(steps + W) < p            # hazard[t] : impact at step t (announced at t-W)
    hits = work = 0
    for t in range(steps):
        imp = hazard[t]
        flag = False
        for (Lg, eps, eta, phi) in tiers:
            if imp and Lg <= W:
                flag |= rng.random() < (1 - eps) * (1 - eta)
            else:
                flag |= rng.random() < phi
        if imp and not flag: hits += 1
        if (not imp) and (not flag): work += 1
    return 100 * hits / steps, 100 * work / steps

if __name__ == '__main__':
    def T1(G, phi=0.02, Lg=1): return (Lg, 0.0, 1 - G, phi)
    T0 = (0, 0.0, 0.0, 0.0)
    cases = [('T1 G=0.90', [T1(0.90)]), ('T1 G=0.50 + T0', [T1(0.50), T0]), ('late + T1 G=0.70', [T1(1.0, 0.02, Lg=2), T1(0.70)]),
             ('three tiers G=0.5,0.6 + late', [T1(0.5, 0.01), T1(0.6, 0.03), T1(1.0, 0.02, Lg=2)])]
    out = []
    for name, tiers in cases:
        g_imp = [((1 - e) * (1 - t)) if Lg <= P_DEF['W'] else f for (Lg, e, t, f) in tiers]
        Gamma = 1 - np.prod([1 - g for g in g_imp]); phi = 1 - np.prod([1 - f for (_, _, _, f) in tiers])
        hit_cf = 100 * P_DEF['p'] * (1 - Gamma); work_cf = 100 * (1 - P_DEF['p']) * (1 - phi)
        hit, work = simulate(tiers, **P_DEF)
        se_hit = 100 * np.sqrt(P_DEF['p'] * (1 - Gamma) * (1 - P_DEF['p'] * (1 - Gamma)) / 2_000_000)
        out.append(dict(name=name, Gamma=float(Gamma), phi=float(phi), hit_sim=hit, hit_cf=hit_cf, work_sim=work, work_cf=work_cf, se_hit=float(se_hit)))
        print(f"{name:32s} Gamma={Gamma:.4f} phi={phi:.4f} hits sim {hit:.4f} vs cf {hit_cf:.4f} (se {se_hit:.4f}); work sim {work:.4f} vs cf {work_cf:.4f}")
    json.dump(out, open('../results/simulate.json', 'w'), indent=1)
