"""POST HOC (added in revision after review R2, findings 5, 7, 8, 12, 17). Not pre-specified.

Derived quantities quoted in the revised text, computed from the committed result files (no new
simulation), plus one new exact computation (grid convergence of the two-threshold latch rows):
 (a) E1 hazard-class split of the gate lambda=0.1, theta=0.2: share of impacts and share of misses
     per class (calm; first announcement of the current storm spell; later in the storm);
 (b) E4 fuse 100/20 at psi=0: paired loss and its s.e., as a fraction of the stress controller's gain;
     the largest GUARD fraction with the fuse 100/20 over psi and its s.e.; censoring of the dwell
     measurements; the fused J against always-WORK by psi;
 (c) E2: GUARD fractions of the sigma rows of Table tab:sigma-mode;
 (d) exact J of the latch rows on quantised chains with N = 2501, 5001, 10001, 20001
     (chain.build_latch_chain / stationary / eval_latch of this package).
Writes ../results/posthoc_r2_derived.json.
"""
import json, time
import numpy as np
from chain import REG, build_latch_chain, sigma_update, stationary, eval_latch

R = '../results/'
e1 = json.load(open(R + 'e1.json'))
e2 = json.load(open(R + 'e2.json'))
e4 = json.load(open(R + 'e4.json'))
out = {'note': 'POST HOC, added in revision after review R2; derived from committed results except (d)'}

# (a) E1 class split
cc = e1['mc_setup']['class_counts']
row = [r for r in e1['mc']['0.1'] if abs(r['theta'] - 0.2) < 1e-12][0]
miss = {'calm': row['frac_missed_calm'] * cc['calm'],
        'storm_first': row['frac_missed_storm_first'] * cc['storm_first'],
        'storm_later': row['frac_missed_storm_later'] * cc['storm_later']}
n_imp = sum(cc.values()); n_miss = sum(miss.values())
out['e1_split_lam0.1_theta0.2'] = dict(
    n_impacts=n_imp, n_missed=n_miss,
    share_of_impacts={k: cc[k] / n_imp for k in cc},
    share_of_misses={k: miss[k] / n_miss for k in miss},
    frac_missed_by_class={'calm': row['frac_missed_calm'], 'storm_first': row['frac_missed_storm_first'],
                          'storm_later': row['frac_missed_storm_later']},
    share_of_misses_unexpected=(miss['calm'] + miss['storm_first']) / n_miss)

# (b) E4
def rows(fuse):
    return [r for r in e4['rows'] if (r['fuse'] == fuse)]
f100 = dict(T_max=100, P=20)
r0 = [r for r in rows(f100) if r['rho'] == 0.0][0]
gain = e2['by_csw']['0.0']['summary']['best_sigma_literal']['J'] - 0.86
out['e4_fuse100_20_psi0'] = dict(paired_loss=-r0['J_minus_nofuse_paired'], paired_loss_se=r0['J_minus_nofuse_paired_se'],
                                 sigma_gain_exact=gain,
                                 loss_frac_of_gain=-r0['J_minus_nofuse_paired'] / gain,
                                 loss_frac_of_gain_se=r0['J_minus_nofuse_paired_se'] / gain)
gmax = max(rows(f100), key=lambda r: r['guard_frac'])
out['e4_fuse100_20_max_guard_frac'] = dict(rho=gmax['rho'], guard_frac=gmax['guard_frac'], se=gmax['guard_frac_se'])
out['e4_fuse100_20_J_by_psi'] = [dict(rho=r['rho'], J=r['J'], J_se=r['J_se'], below_always_work=r['J'] < 0.86,
                                      z_vs_always_work=(r['J'] - 0.86) / r['J_se']) for r in rows(f100)]
out['e4_nofuse_dwell_censoring'] = [dict(rho=r['rho'], n=r['dwell_after_storm_end']['n'],
                                         n_censored=r['dwell_after_storm_end']['n_censored'],
                                         mean=r['dwell_after_storm_end']['mean'], p95=r['dwell_after_storm_end']['p95'])
                                    for r in rows(None)]
out['e4_fuse_settings'] = sorted({(r['fuse']['T_max'], r['fuse']['P']) for r in e4['rows'] if r['fuse']})
out['e4_n_fused_cells'] = sum(1 for r in e4['rows'] if r['fuse'])

# (c) E2 GUARD fractions
S0 = e2['by_csw']['0.0']['sigma']; S1 = e2['by_csw']['1.0']['sigma']
def gf(S, lam, kind):
    return S[lam][kind]['guard_frac'] if kind in S[lam] else None
out['e2_guard_frac'] = {k: {kk: (vv.get('guard_frac') if isinstance(vv, dict) else None)
                            for kk, vv in S0[k].items() if isinstance(vv, dict)} for k in S0}
out['e2_guard_frac_csw1'] = {k: {kk: (vv.get('guard_frac') if isinstance(vv, dict) else None)
                                 for kk, vv in S1[k].items() if isinstance(vv, dict)} for k in S1}

# (d) latch grid convergence (exact)
t0 = time.time()
lat = []
for csw, lam, ton, toff in [(0.0, 0.1, None, None), (0.0, 0.5, 0.5315, 0.0064), (1.0, 0.1, 0.2302, 0.0576)]:
    prm = dict(REG); prm['c_sw'] = csw
    if ton is None:
        b = e2['by_csw']['0.0']['summary']['best_sigma_literal']; ton, toff = b['theta_on'], b['theta_off']
    else:
        S = e2['by_csw'][str(csw)]['sigma'][str(lam)]
        cand = [v for v in S.values() if isinstance(v, dict) and 'theta_on' in v and abs(v['theta_on'] - ton) < 1e-4]
        ton, toff = cand[0]['theta_on'], cand[0]['theta_off']
    vals = {}
    for N in (2501, 5001, 10001, 20001):
        P, D, Z, LP = build_latch_chain(prm, sigma_update(lam), N, ton, toff, 'literal')
        pi = stationary(P, tol=1e-13)
        vals[str(N)] = eval_latch(prm, pi, D, Z, LP)['J']
    v = list(vals.values())
    lat.append(dict(c_sw=csw, lam=lam, theta_on=ton, theta_off=toff, J_by_N=vals,
                    max_change_halving_doubling_around_10001=max(abs(vals['5001'] - vals['10001']), abs(vals['20001'] - vals['10001'])),
                    max_change_any=max(v) - min(v)))
    print('latch', csw, lam, vals, flush=True)
out['latch_grid_convergence'] = lat
out['latch_grid_seconds'] = time.time() - t0
json.dump(out, open(R + 'posthoc_r2_derived.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ('e2_guard_frac', 'e2_guard_frac_csw1')}, indent=1, default=float))
