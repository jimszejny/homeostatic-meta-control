"""E4: stress lock-in through behaviour-mediated self-evidence, and a dwell-limit fuse.
sigma input u_t = max(a_t, e_t), e_t ~ Bern(rho), while the committed mode l_t is GUARD; u_t = a_t in
WORK. Controller = best (lambda, theta_on, theta_off) of E2 at c_sw=0 (literal timing), unchanged.
Fuse (circuit breaker with half-open state): a GUARD stint may last T_max committed steps; the step
that would be GUARD step T_max+1 is instead the first of P forced-WORK probe steps, and sigma is
reset to 0 on that step. Vectorised Monte Carlo over (config x chain); exact quantised-chain
cross-check for the no-fuse rows."""
import json, time
import numpy as np
from chain import *

t0 = time.time()
base = dict(REG)
e2 = json.load(open('../results/e2.json'))
best = e2['by_csw']['0.0']['summary']['best_sigma_literal']
LAM = float(e2['by_csw']['0.0']['summary']['best_lambda_literal'])
TH_ON, TH_OFF = float(best['theta_on']), float(best['theta_off'])
print('config', LAM, TH_ON, TH_OFF, flush=True)

RHOS_BRIEF = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
RHOS_EXTRA = [0.07, 0.085, 0.1, 0.115, 0.13, 0.15, 0.25, 0.4, 0.5]      # added sweep (same script, not post hoc)
RHOS = sorted(set(RHOS_BRIEF + RHOS_EXTRA))
FUSES = [None, (50, 10), (50, 20), (100, 10), (100, 20), (200, 10), (200, 20)]
CH = 64                     # chains per config (chain j uses the same random streams in every config: CRN)
T, BURN = 60000, 4000
cfg = [(r, f) for r in RHOS for f in FUSES]
nC = len(cfg) * CH
rho = np.repeat([c[0] for c in cfg], CH)
tmax = np.repeat([c[1][0] if c[1] else 10**9 for c in cfg], CH)
pwin = np.repeat([c[1][1] if c[1] else 0 for c in cfg], CH)

rng = np.random.default_rng(404)
Q = Qmat(base['q01'], base['q10'])
ps = np.array([base['p0'], base['p1']])
imp = impact_prob_given_z(base)
k = kG(base['G'], base['h'])
wz = 1 - imp * k
pz = stationary_z(base['q01'], base['q10'])
nK = len(cfg)
tile = lambda x: np.tile(x, nK)
z = tile(rng.random(CH) < pz[1]).astype(np.int8)
sig = np.zeros(nC); l = np.zeros(nC, np.int8); dwell = np.zeros(nC, np.int64); probe = np.zeros(nC, np.int64)
last_a = np.full(nC, -10**9, np.int64)
# accumulators (after burn-in)
acc_r = np.zeros(nC); acc_g = np.zeros(nC); acc_sw = np.zeros(nC); acc_hits = np.zeros(nC); n = 0
acc_trips = np.zeros(nC)
max_d = np.zeros(nC, np.int64)              # max over GUARD steps of (t - last real announcement)
cnt = np.zeros(nC, np.int64); max_cnt = np.zeros(nC, np.int64)   # GUARD steps counted since last real announcement
# dwell after storm end: consecutive GUARD steps from the storm-end step
in_meas = np.zeros(nC, bool); meas_start = np.zeros(nC, np.int64)
dw_records = [[] for _ in range(len(cfg))]
cfg_of = np.repeat(np.arange(len(cfg)), CH)
for t in range(T):
    # 1. commit a mode
    lprev = l.copy()
    inprobe = probe > 0
    lat = np.where(sig >= TH_ON - 1e-12, 1, np.where(sig < TH_OFF - 1e-12, 0, lprev))
    lnew = np.where(inprobe, 0, lat).astype(np.int8)
    probe = np.where(inprobe, probe - 1, probe)
    dwell = np.where(lnew == 1, dwell + 1, 0)
    trip = (lnew == 1) & (dwell > tmax)
    lnew = np.where(trip, 0, lnew).astype(np.int8)
    sig = np.where(trip, 0.0, sig)
    probe = np.where(trip, pwin - 1, probe)
    dwell = np.where(trip, 0, dwell)
    l = lnew
    # 2. announcements and self-evidence
    a = tile(rng.random(CH)) < ps[z]
    e = (tile(rng.random(CH)) < rho) & (l == 1)
    u = a | e
    if t >= BURN:
        acc_r += np.where(l == 1, base['r_g'], wz[z]); acc_g += l; acc_sw += (l != lprev)
        acc_hits += np.where(l == 1, 0, imp[z] * (1 - base['G'])); acc_trips += trip; n += 1
        d = np.where(l == 1, t - last_a, 0)
        max_d = np.maximum(max_d, d)
        cnt = cnt + (l == 1)
        max_cnt = np.maximum(max_cnt, cnt)
    cnt = np.where(a, 0, cnt)
    last_a = np.where(a, t, last_a)
    # dwell-after-storm-end bookkeeping (on z_t -> z_{t+1})
    sig = (1 - LAM) * sig + LAM * u
    U = tile(rng.random(CH))
    znew = np.where(z == 0, (U < base['q01']).astype(np.int8), (U >= base['q10']).astype(np.int8))
    ended = (z == 1) & (znew == 0)
    if t >= BURN:
        # a storm ended: start measuring at step t+1; close running measurements when mode is WORK
        done = in_meas & (l == 0)
        for i in np.nonzero(done)[0]:
            dw_records[cfg_of[i]].append((t - meas_start[i], 0))
        in_meas &= ~done
        new = ended & ~in_meas
        meas_start = np.where(new, t + 1, meas_start); in_meas |= new
    z = znew
for i in np.nonzero(in_meas)[0]:
    dw_records[cfg_of[i]].append((T - meas_start[i], 1))   # censored at end of run

res = dict(config=dict(lam=LAM, theta_on=TH_ON, theta_off=TH_OFF), mc=dict(chains_per_config=CH, T=T, burn=BURN, seed=404),
           mean_field=dict(brief_condition='rho >= theta_off', theta_off=TH_OFF,
                           with_calm_announcements='1-(1-p0)(1-rho) >= theta_off  <=>  rho >= (theta_off-p0)/(1-p0) = %.4f' % ((TH_OFF - base['p0']) / (1 - base['p0']))),
           rows=[])
for ci, (r, f) in enumerate(cfg):
    sl = slice(ci * CH, (ci + 1) * CH)
    def ms(x):
        v = x[sl] / n
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(CH))
    rec = np.array(dw_records[ci]) if dw_records[ci] else np.zeros((0, 2))
    dws = rec[:, 0] if len(rec) else np.array([0])
    row = dict(rho=r, fuse=None if f is None else dict(T_max=f[0], P=f[1]))
    for name, arr in (('J', acc_r), ('guard_frac', acc_g), ('switches_per_step', acc_sw), ('hits_per_step', acc_hits), ('trips_per_step', acc_trips)):
        row[name], row[name + '_se'] = ms(arr)
    row['dwell_after_storm_end'] = dict(n=int(len(rec)), mean=float(dws.mean()), p95=float(np.percentile(dws, 95)),
                                        max=float(dws.max()), n_censored=int(rec[:, 1].sum()) if len(rec) else 0,
                                        frac_zero=float(np.mean(dws == 0)))
    row['max_guard_steps_after_last_real_announcement'] = int(max_d[sl].max())      # time span
    row['max_guard_count_after_last_real_announcement'] = int(max_cnt[sl].max())     # number of GUARD steps
    row['J_chain'] = acc_r[sl] / n
    if f is not None:
        row['span_bound_Tmax_plus_P_minus_1'] = f[0] + f[1] - 1
        row['span_bound_holds'] = bool(max_d[sl].max() <= f[0] + f[1] - 1)
        row['count_bound_T_max_holds'] = bool(max_cnt[sl].max() <= f[0])
        row['naive_span_bound_T_max_holds'] = bool(max_d[sl].max() <= f[0])
    res['rows'].append(row)
    print(f"rho={r:<6} fuse={f}: J={row['J']:.5f}±{row['J_se']:.5f} guard={row['guard_frac']:.4f} "
          f"dwell mean={row['dwell_after_storm_end']['mean']:.1f} p95={row['dwell_after_storm_end']['p95']:.0f} "
          f"max={row['dwell_after_storm_end']['max']:.0f} cens={row['dwell_after_storm_end']['n_censored']} "
          f"maxd={row['max_guard_steps_after_last_real_announcement']}", flush=True)
# paired (CRN) fuse cost and no-fuse differences
for row in res['rows']:
    ref = [x for x in res['rows'] if x['rho'] == row['rho'] and x['fuse'] is None][0]
    d = row['J_chain'] - ref['J_chain']
    row['J_minus_nofuse_paired'] = float(d.mean()); row['J_minus_nofuse_paired_se'] = float(d.std(ddof=1) / np.sqrt(CH))
for row in res['rows']:
    if row['fuse'] is not None:
        print('rho', row['rho'], row['fuse'], 'paired diff vs no fuse %.6f +- %.6f' % (row['J_minus_nofuse_paired'], row['J_minus_nofuse_paired_se']),
              'count<=Tmax', row['count_bound_T_max_holds'], 'span<=Tmax+P-1', row['span_bound_holds'], 'span<=Tmax', row['naive_span_bound_T_max_holds'])
print(f'MC done {time.time()-t0:.0f}s', flush=True)

# exact cross-check (no fuse) on the quantised (sigma, l, z) chain
N = 5001
xc = []
for r in [0.0, 0.05, 0.1, 0.2, 0.3]:
    P, D, Z, LP = build_latch_chain(base, sigma_update(LAM), N, TH_ON, TH_OFF, 'literal', rho=r)
    pi = stationary(P, tol=1e-12)
    ev = eval_latch(base, pi, D, Z, LP)
    mc = [x for x in res['rows'] if x['rho'] == r and x['fuse'] is None][0]
    xc.append(dict(rho=r, exact_J=ev['J'], exact_guard=ev['guard_frac'], mc_J=mc['J'], mc_J_se=mc['J_se'],
                   mc_guard=mc['guard_frac'], mc_guard_se=mc['guard_frac_se'],
                   z_J=(mc['J'] - ev['J']) / mc['J_se'], z_guard=(mc['guard_frac'] - ev['guard_frac']) / max(mc['guard_frac_se'], 1e-12)))
    print('xcheck', xc[-1], flush=True)
res['crosscheck_exact_vs_mc'] = xc
res['linear_toy'] = dict(
    recursion='sigma_{t+1} = (1-lambda) sigma_t + lambda (a_t + k sigma_t) = (1 - lambda + lambda k) sigma_t + lambda a_t',
    unstable_iff='|1-lambda+lambda k| >= 1  <=>  k >= 1 (for 0<lambda<=1, k>=0)',
    continuous='d sigma/dt = -lambda_c sigma + k_c sigma + g: unstable iff k_c >= lambda_c (feedback gain >= leak rate); '
               'Euler step dt: lambda = lambda_c dt, lambda k = k_c dt  =>  k = k_c/lambda_c')
res['wall_s'] = time.time() - t0
json.dump(strip_chain(res), open('../results/e4.json', 'w'), indent=1, default=float)
print(f'E4 done {time.time()-t0:.1f}s')
