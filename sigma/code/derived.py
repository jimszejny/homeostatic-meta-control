"""Derived quantities quoted in the paper text (each computed here from results/*.json or from the
fixed parameters in chain.REG) -> results/derived.json. Keys are cited in paper/SIGMA_NOTES.txt."""
import json
import numpy as np
from chain import REG, kG, stationary_z, Qmat
R = '../results/'
e1 = json.load(open(R + 'e1.json')); e2 = json.load(open(R + 'e2.json')); e3 = json.load(open(R + 'e3.json'))
e4 = json.load(open(R + 'e4.json')); e5 = json.load(open(R + 'e5.json'))
p = REG
d = {}
d['params'] = dict(p0=p['p0'], p1=p['p1'], q01=p['q01'], q10=p['q10'], h=p['h'], Gamma=p['G'], r_g=p['r_g'], c_sw=[0.0, 1.0],
                   L=p['L'], W=p['W'], gamma=p['gamma'], psig=p['psig'], c_r=12.4787,
                   storm_fraction=float(stationary_z(p['q01'], p['q10'])[1]), pbar=0.05,
                   inv_q01=1 / p['q01'], inv_q10=1 / p['q10'], inv_p1=1 / p['p1'], inv_p0=1 / p['p0'],
                   inv_q01_plus_q10=1 / (p['q01'] + p['q10']),
                   h_Gamma=kG(p['G']), p0_hG=p['p0'] * kG(p['G']), p1_hG=p['p1'] * kG(p['G']), one_minus_rg=1 - p['r_g'],
                   one_over_lambda_best=1 / 0.1, one_over_lambda_005=1 / 0.05,
                   lambdas=e2['lambdas'], G_list=e3['G_list'], rg_list=e3['rg_list'],
                   rhos=sorted(set(r['rho'] for r in e4['rows'])), Tdw=[50, 100, 200], Tpr=[10, 20],
                   persistence_scales=[0.25, 0.5, 1, 2, 4, 8], second_eigenvalue=1 - p['q01'] - p['q10'])
S0 = e2['by_csw']['0.0']; S1 = e2['by_csw']['1.0']
g0 = S0['summary']['J_Bayes'] - S0['summary']['J_bestfixed']
g1 = S1['summary']['J_Bayes'] - S1['summary']['J_bestfixed']
bs0 = S0['summary']['best_sigma_literal']; bs1 = S1['summary']['best_sigma_literal']
d['e2'] = dict(bayes_gain_c0=g0, bayes_gain_c1=g1, oracle_gain_c0=S0['C-oracle']['J'] - 0.86,
               sigma_gain_c0=bs0['J'] - 0.86, sigma_gain_c1=bs1['J'] - 0.86,
               frac_c0_pct=100 * bs0['frac_gain'], frac_c1_pct=100 * bs1['frac_gain'],
               frac_lam1_pct=100 * S0['sigma']['1']['literal_best']['frac_gain'],
               frac_lam0005_pct=100 * S0['sigma']['0.005']['literal_best']['frac_gain'],
               matched_best_frac_pct=100 * S0['sigma'][str(S0['summary']['best_lambda_matched'])]['matched_single']['frac_gain'],
               bayes_over_oracle_gain_pct=100 * g0 / (S0['C-oracle']['J'] - 0.86),
               myopic_bayes_loss_c1=0.86 - S1['C-Bayes-myopic']['J'],
               hits_reduction_bayes_pct=100 * (1 - S0['C-Bayes']['hits'] / S0['C-W']['hits']),
               hits_reduction_sigma_pct=100 * (1 - bs0['hits'] / S0['C-W']['hits']),
               relative_gain_bayes_pct=100 * g0 / 0.86, relative_gain_oracle_pct=100 * (S0['C-oracle']['J'] - 0.86) / 0.86)
# E1 at selected points
def e1pt(lam, th):
    r = [x for x in e1['exact'][str(lam)] if abs(x['theta'] - th) < 1e-9][0]
    return dict(frac_missed_pct=100 * r['frac_missed'], hits_G1=r['hits_G1'], hits_G08=r['hits_G08'])
d['e1'] = {'%g_%g' % (l, t): e1pt(l, t) for l, t in [(0.1, 0.2), (0.1, 0.12), (0.05, 0.1), (0.02, 0.04), (0.2, 0.3), (0.5, 0.6), (0.3, 0.5)]}
mc = {}
for lam, th in [(0.1, 0.2), (0.1, 0.12), (0.05, 0.1), (0.5, 0.6), (0.02, 0.04)]:
    r = [x for x in e1['mc'][str(lam)] if abs(x['theta'] - th) < 1e-9][0]
    mc['%g_%g' % (lam, th)] = {k + '_pct': 100 * r[k] for k in ('frac_missed', 'frac_missed_calm', 'frac_missed_storm_first', 'frac_missed_storm_later')}
d['e1_mc_pct'] = mc
d['e1_threshold_bound'] = {'%g_%g' % (l, t): (t - l) / (1 - l) for l, t in [(0.1, 0.2), (0.05, 0.1), (0.1, 0.12)]}
# E3
d['e3'] = dict(window_cells=sum(c['analytic_condition_switching_useful'] for c in e3['phase']),
               gain_cells=sum(c['Bayes_gain'] > 1e-9 for c in e3['phase']), n_cells=len(e3['phase']),
               frac_range_pct=[100 * min(c['frac_captured'] for c in e3['phase'] if c['frac_captured'] is not None and c['Bayes_gain'] > 1e-3),
                               100 * max(c['frac_captured'] for c in e3['phase'] if c['frac_captured'] is not None)],
               G065_rg09_threshold=(1 - 0.9) / kG(0.65))
Q2 = np.linalg.matrix_power(Qmat(p['q01'], p['q10']), 2)
d['e3']['calm_floor_note'] = 'see REPORT'
# E4
rows = e4['rows']
def row(r, f):
    return [x for x in rows if x['rho'] == r and x['fuse'] == f][0]
nf = {str(r): row(r, None) for r in d['params']['rhos']}
f100 = {str(r): row(r, {'T_max': 100, 'P': 20}) for r in d['params']['rhos']}
d['e4'] = dict(theta_off=e4['config']['theta_off'], theta_on=e4['config']['theta_on'],
               meanfield_brief=e4['config']['theta_off'], meanfield_calm=(e4['config']['theta_off'] - p['p0']) / (1 - p['p0']),
               sigma_std_rho02=float(np.sqrt(0.1 / (2 - 0.1) * (1 - 0.98 * 0.8) * (0.98 * 0.8))),
               fuse_cost_c0={'%d_%d' % (f['T_max'], f['P']): -row(0.0, f)['J_minus_nofuse_paired'] for f in
                             [{'T_max': a, 'P': b} for a in (50, 100, 200) for b in (10, 20)]},
               fuse_cost_pct_of_gain={'%d_%d' % (a, b): -100 * row(0.0, {'T_max': a, 'P': b})['J_minus_nofuse_paired'] / (bs0['J'] - 0.86)
                                      for a in (50, 100, 200) for b in (10, 20)},
               max_guard_fuse100_20=max(x['guard_frac'] for x in f100.values()),
               sigma_gain_c0=bs0['J'] - 0.86)
# E5
d['e5'] = dict(avg_diff_full=e5['C-Bayes']['stationary_avg_full'] - e5['C-W']['stationary_avg_full'],
               avg_diff_shortcut=e5['C-Bayes']['stationary_avg_shortcut'] - e5['C-W']['stationary_avg_shortcut'],
               cw_shortcut_avg_minus_prop=e5['C-W']['stationary_avg_shortcut'] - e5['validation_p0_eq_p1']['0.8']['closed_form_prop1'],
               max_bisect_vs_sweep=max(abs(c['bisect'] - c['sweep_interp']) for n in ('C-W', 'C-Bayes') for c in e5[n]['bisect_checks']),
               validation_max_err=max(max(abs(v['shortcut'] - v['closed_form_prop1']), abs(v['bisect_full'] - v['closed_form_prop1']))
                                      for v in e5['validation_p0_eq_p1'].values()))
d['e2']['frac_lam05_single_pct'] = 100 * S0['sigma']['0.5']['literal_single']['frac_gain']
d['e2']['frac_lam05_latch_pct'] = 100 * S0['sigma']['0.5']['literal_best']['frac_gain']
d['params']['calm_hazard_share'] = 0.8 * p['p0'] / 0.05
d['params']['window_rg_min_Gamma1'] = 1 - p['p1']
fused = [x for x in rows if x['fuse'] is not None]
d['e4']['n_fused'] = len(fused)
d['e4']['n_naive_span_violations'] = sum(not x['naive_span_bound_T_max_holds'] for x in fused)
d['e4']['max_naive_excess'] = max(x['max_guard_steps_after_last_real_announcement'] - x['fuse']['T_max'] for x in fused)
d['e4']['all_count_and_span_bounds_hold'] = all(x['count_bound_T_max_holds'] and x['span_bound_holds'] for x in fused)
d['e4']['max_guard_frac_se'] = max(x['guard_frac_se'] for x in rows); d['e4']['max_J_se'] = max(x['J_se'] for x in rows)
d['e4']['max_abs_z'] = max(max(abs(x['z_J']), abs(x['z_guard'])) for x in e4['crosscheck_exact_vs_mc'])
d['e2']['max_abs_paired_z'] = max(abs(x['literal']['diff_vs_W_mc_minus_exact']['z']) if 'literal' in x else abs(x['diff_vs_W_mc_minus_exact']['z'])
    for c in ('0.0', '1.0') for x in e2['by_csw'][c]['crosscheck'] if x.get('name') != 'C-W'
    and ('literal' not in x or x['literal']['diff_vs_W_mc_minus_exact']['paired_se'] > 0))
g = e2['grid_convergence']; d['e2']['grid_max_change_vs_10001'] = max(abs(r[k] - g[2][k]) for r in (g[1], g[3]) for k in r if k != 'N')
d['e2']['oracle_share_c0'] = (S0['C-oracle']['J'] - 0.86) / g0
d['mc_lengths'] = dict(e1_steps_after_burn=e1['mc_setup']['T'] - e1['mc_setup']['burn'], e4_steps_after_burn=e4['mc']['T'] - e4['mc']['burn'],
                       e2_test_steps_after_burn=e2['mc_setup']['test']['T'] - e2['mc_setup']['test']['burn'])
json.dump(d, open(R + 'derived.json', 'w'), indent=1, default=float)
print(json.dumps(d, indent=1, default=float)[:6000])
