"""E1: cortisol must not gate adrenaline.
Always WORK. Raw gate: the reflex is engaged on every announced impact (catch rate G).
Filtered gate C-gate(lambda, theta): reflex engaged at impact step t only if sigma_t >= theta,
sigma_{t+1}=(1-lambda)sigma_t+lambda a_t (sigma_t contains a_{t-1}, the announcement of the
impact at t). Engaged reflex dodges an impact w.p. G, never fires without an impact (phi=0).
Exact part: quantised (sigma,z) chain, N grid points, closed-form miss condition
   miss at impact step t  <=>  (1-lambda) sigma_{t-1} + lambda < theta.
MC part: hazard-class split (calm / first of a storm / later in a storm) and cross-check."""
import json, time, sys
import numpy as np
from chain import *

t0 = time.time()
prm = dict(REG)
N = 10001
LAMS = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
THETAS = np.round(np.linspace(0.0, 0.6, 121), 4)
g = np.linspace(0, 1, N)
ps = np.array([prm['p0'], prm['p1']])
pz = stationary_z(prm['q01'], prm['q10'])
pbar = float(pz @ ps)

res = dict(N_grid=N, thetas=THETAS.tolist(), lambdas=LAMS, pbar=pbar, exact={}, mc={})
for lam in LAMS:
    P = build_filter_chain(prm, sigma_update(lam), N, 'literal')
    pi = stationary(P).reshape(2, N)       # law of (sigma_{t-1}, z_{t-1})
    imp_mass = pi * ps[:, None]            # joint with a_{t-1}=1 (impact at t)
    sig_at_imp = (1 - lam) * g + lam       # sigma_t at the impact step
    sig_at_nonimp = (1 - lam) * g
    nonimp_mass = pi * (1 - ps[:, None])
    rows = []
    for th in THETAS:
        miss = float(imp_mass[:, sig_at_imp < th - 1e-12].sum())
        miss_calm = float(imp_mass[0, sig_at_imp < th - 1e-12].sum())
        on_nonimp = float(nonimp_mass[:, sig_at_nonimp >= th - 1e-12].sum())
        rows.append(dict(theta=float(th), frac_missed=miss / pbar,
                         frac_missed_calm=miss_calm / float(imp_mass[0].sum()),
                         frac_missed_storm=(miss - miss_calm) / float(imp_mass[1].sum()),
                         hits_G1=100 * miss, hits_G08=100 * (pbar - 0.8 * (pbar - miss)),
                         gate_on_nonimpact_per100=100 * on_nonimp))
    res['exact'][str(lam)] = rows
    print(f'lam={lam}: theta=0.1 missed {rows[20]["frac_missed"]:.4f}, theta=0.3 {rows[60]["frac_missed"]:.4f}',
          f'{time.time()-t0:.1f}s', flush=True)
res['raw_gate'] = dict(hits_G1=0.0, hits_G08=100 * pbar * 0.2, frac_missed=0.0)

# ---- Monte Carlo: class split + cross-check --------------------------------------------------
C, T, burn = 400, 50000, 2000
z, a, _ = simulate_regime(prm, C, T, seed=101)
imp = np.zeros_like(a); imp[:, 1:] = a[:, :-1]           # impact at t iff a_{t-1}
zprev = np.zeros_like(z); zprev[:, 1:] = z[:, :-1]       # regime of the announcement
# storm sojourn bookkeeping on the announcement index u=t-1
ar = np.arange(T)[None, :]
start = (z == 1) & np.concatenate([np.ones((C, 1), bool), z[:, :-1] == 0], 1)
sidx = np.maximum.accumulate(np.where(start, ar, 0), axis=1)
cs = np.cumsum(a * (z == 1), axis=1)
base = np.where(sidx > 0, np.take_along_axis(cs, np.maximum(sidx - 1, 0), 1), 0)
first_in_storm = (z == 1) & (a == 1) & ((cs - 1 - base) == 0)      # indexed by u
first_at_t = np.zeros_like(first_in_storm); first_at_t[:, 1:] = first_in_storm[:, :-1]
mask = np.zeros((C, T), bool); mask[:, burn:] = True
cls = {'calm': (imp == 1) & (zprev == 0) & mask,
       'storm_first': (imp == 1) & (zprev == 1) & first_at_t & mask,
       'storm_later': (imp == 1) & (zprev == 1) & ~first_at_t & mask}
n_imp = int(((imp == 1) & mask).sum())
res['mc_setup'] = dict(C=C, T=T, burn=burn, seed=101, n_impacts=n_imp,
                       class_counts={k: int(v.sum()) for k, v in cls.items()})
MC_LAMS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.005]
MC_TH = np.round(np.linspace(0.0, 0.6, 31), 4)
for lam in MC_LAMS:
    lit, _ = sigma_paths(a, lam)
    rows = []
    for th in MC_TH:
        off = lit < th - 1e-12
        missed = (imp == 1) & mask & off
        per_chain = missed.sum(1) / ((imp == 1) & mask).sum(1)
        hits_chain = 100 * missed.sum(1) / mask.sum(1)
        d = dict(theta=float(th), frac_missed=float(missed.sum() / n_imp),
                 frac_missed_se=float(per_chain.std(ddof=1) / np.sqrt(C)),
                 hits_G1=float(hits_chain.mean()), hits_G1_se=float(hits_chain.std(ddof=1) / np.sqrt(C)))
        for k, v in cls.items():
            d['frac_missed_' + k] = float((v & off).sum() / max(v.sum(), 1))
        rows.append(d)
    res['mc'][str(lam)] = rows
    del lit
# cross-check table exact vs MC
xc = []
for lam, th in [(0.5, 0.6), (0.2, 0.3), (0.1, 0.2), (0.05, 0.1), (0.02, 0.04), (0.005, 0.02)]:
    ex = [r for r in res['exact'][str(lam)] if abs(r['theta'] - th) < 1e-9][0]
    mc = [r for r in res['mc'][str(lam)] if abs(r['theta'] - th) < 1e-9][0]
    xc.append(dict(lam=lam, theta=th, exact=ex['frac_missed'], mc=mc['frac_missed'],
                   mc_se=mc['frac_missed_se'], z=(mc['frac_missed'] - ex['frac_missed']) / max(mc['frac_missed_se'], 1e-12)))
    print('xcheck', xc[-1])
res['crosscheck'] = xc
res['wall_s'] = time.time() - t0
json.dump(res, open('../results/e1.json', 'w'), indent=1)
print(f'E1 done {time.time()-t0:.1f}s')
