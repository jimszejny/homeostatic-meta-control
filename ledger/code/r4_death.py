"""R4: death state on the latency chain. c*_r(omega) for delta in {0,0.05,0.2}, several tiers,
death booked at 0 or at D=1 (entry price omega*gamma), off neutral (L0) or priced (L1)."""
import json, os
import numpy as np
from scipy.optimize import brentq
from chain import Chain, CR, closed_cstar, admissibility

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
g = 0.95
OMEGAS = [0.0, 0.5, 1.0, 2.0, 4.0]
DELTAS = [0.0, 0.05, 0.2]
CONFIGS = [('planner', 0.0, 0.0), ('T1 G=0.50', 0.5, 0.02), ('T1 G=0.80', 0.8, 0.02),
           ('T1 G=0.95', 0.95, 0.02), ('reflex', 1.0, 0.0)]
BOOK = [('death@0, off L0', 0.0, 0.0), ('death@1, off L0', 1.0, 0.0),
        ('death@0, off L1', 0.0, 1.0), ('death@1, off L1', 1.0, 1.0)]


def via_for(ch, Dd, Do):
    return ch.via('PB', D_off=Do, D_death=Dd)


def cont_optimal(ch, cr, w, via):
    V, Q, pol = ch.solve(ch.reward(cr, w, via))
    for i, x in enumerate(ch.live):
        row = ch.rows[pol[i]]
        if x[5] == 1 and row['kind'] != 'comply':
            return False
        if x[5] == 0 and row['a'] != 0:
            return False
    return True


rows = []
for cname, G, phi in CONFIGS:
    for d in DELTAS:
        ch = Chain(Gamma=G, phi=phi, delta=d)
        for bname, Dd, Do in BOOK:
            via = via_for(ch, Dd, Do)
            vals = []
            for w in OMEGAS:
                cs = ch.cstar(w, via)
                cf, c0, beta, Kd, Md = closed_cstar(Gamma=G, phi=phi, delta=d, omega=w,
                                                    off_price=g * Do, death_price=g * Dd)
                gain, Vbar = admissibility(Gamma=G, phi=phi, delta=d, omega=w,
                                           off_price=g * Do, death_price=g * Dd)
                cont = cont_optimal(ch, cs + 1e-6, w, via)
                vals.append(dict(omega=w, cstar=cs, closed=cf, err=abs(cs - cf), adm_gain=gain,
                                 Vbar=Vbar, continuation_optimal=cont,
                                 verdict='resist' if CR < cs else 'comply'))
            c = [v['cstar'] for v in vals]
            slope = np.polyfit(OMEGAS, c, 1)[0]
            r = dict(config=cname, Gamma=G, phi=phi, delta=d, booking=bname, beta_closed=beta,
                     cstar0_closed=c0, K_delta=Kd, M_delta=Md, fitted_slope=slope,
                     spread=max(c) - min(c), max_err=max(v['err'] for v in vals), values=vals)
            rows.append(r)
            print(f"{cname:10s} d={d:4.2f} {bname:16s} c*=" + ' '.join(f"{x:8.4f}" for x in c) +
                  f" beta={beta:+.5f} slope={slope:+.5f} err={r['max_err']:.1e} "
                  f"adm={min(v['adm_gain'] for v in vals):.3f} cont={all(v['continuation_optimal'] for v in vals)} "
                  f"v={''.join(v['verdict'][0] for v in vals)}")

# ---- extended sweep (labelled): where affinity in omega fails because the work policy changes
ext = []
for cname, G, phi in [('planner', 0.0, 0.0), ('T1 G=0.50', 0.5, 0.02)]:
    d = 0.2
    ch = Chain(Gamma=G, phi=phi, delta=d)
    via = via_for(ch, 1.0, 0.0)
    f = lambda w: admissibility(Gamma=G, phi=phi, delta=d, omega=w, death_price=g)[0]
    wb = brentq(f, 0.0, 1e4)
    pts = []
    for w in sorted(set(list(np.round(np.linspace(0, 1.6 * wb, 33), 6)) + [wb * (1 - 1e-6), wb * (1 + 1e-6)])):
        cs = ch.cstar(w, via, lo=-50.0)
        cf = closed_cstar(Gamma=G, phi=phi, delta=d, omega=w, death_price=g)[0]
        V, Q, pol = ch.solve(ch.reward(cs + 1e-6, w, via))
        frac_work = np.mean([ch.rows[pol[i]]['a'] == 0 for i, x in enumerate(ch.live) if x[5] == 0])
        pts.append(dict(omega=float(w), cstar=cs, closed=cf, err=abs(cs - cf), frac_work_commit=float(frac_work)))
    ext.append(dict(config=cname, delta=d, booking='death@1, off L0', omega_break_closed=wb, points=pts))
    print(f"[extended] {cname} delta=0.2 death@1: admissibility fails at omega={wb:.6f}")
    for q in pts:
        if abs(q['omega'] - wb) / wb < 0.2 or q['omega'] == 0:
            print(f"   w={q['omega']:10.4f} c*={q['cstar']:9.5f} closed={q['closed']:9.5f} err={q['err']:.1e} work={q['frac_work_commit']:.2f}")

json.dump(dict(rows=rows, extended=ext), open(os.path.join(OUT, 'r4.json'), 'w'), indent=1)
