"""Print the result tables (markdown) from results/*.json -> results/tables.md."""
import json
R = '../results/'
e1 = json.load(open(R + 'e1.json')); e2 = json.load(open(R + 'e2.json')); e3 = json.load(open(R + 'e3.json'))
e4 = json.load(open(R + 'e4.json')); e5 = json.load(open(R + 'e5.json')); s0 = json.load(open(R + 'step0.json'))
L = []
P = L.append
P('## Step 0 (exact)\n\n| config | V_start | c*_r bisection | c*_r closed form | hits/100 | work/100 |\n|---|---|---|---|---|---|')
for k, v in s0.items():
    if isinstance(v, dict) and 'V_start' in v:
        P(f"| {k} | {v['V_start']:.6f} | {v['cstar_bisect']:.10f} | {v['cstar_closed']:.10f} | {v['hits_per100']:.4f} | {v['work_per100']:.4f} |")
P('\n## E1: filtered gate, fraction of impacts missed by the gate (exact, quantised chain N=10001)\n')
ths = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
P('| lambda | ' + ' | '.join('th=%g' % t for t in ths) + ' |'); P('|---' * (len(ths) + 1) + '|')
for lam in e1['lambdas']:
    rows = {round(r['theta'], 4): r for r in e1['exact'][str(lam)]}
    P(f'| {lam} | ' + ' | '.join('%.4f' % rows[round(t, 4)]['frac_missed'] for t in ths) + ' |')
P('\nHits/100 at G=1 = 100 * pbar * frac_missed (pbar=%.4f); raw gate: 0 (G=1), %.2f (G=0.8).' % (e1['pbar'], e1['raw_gate']['hits_G08']))
P('\nMC class split (seed 101, %d impacts; counts %s): fraction missed\n' % (e1['mc_setup']['n_impacts'], e1['mc_setup']['class_counts']))
P('| lambda | theta | all (exact) | all (MC, se) | calm | first of storm | later in storm |\n|---|---|---|---|---|---|---|')
for lam, th in [(0.5, 0.6), (0.2, 0.3), (0.1, 0.2), (0.1, 0.12), (0.05, 0.1), (0.02, 0.04), (0.02, 0.1), (0.005, 0.02)]:
    ex = [r for r in e1['exact'][str(lam)] if abs(r['theta'] - th) < 1e-9][0]
    mc = [r for r in e1['mc'][str(lam)] if abs(r['theta'] - th) < 1e-9]
    if not mc: continue
    mc = mc[0]
    P(f"| {lam} | {th} | {ex['frac_missed']:.4f} | {mc['frac_missed']:.4f} ({mc['frac_missed_se']:.4f}) | {mc['frac_missed_calm']:.4f} | {mc['frac_missed_storm_first']:.4f} | {mc['frac_missed_storm_later']:.4f} |")
for csw in ('0.0', '1.0'):
    S = e2['by_csw'][csw]; sm = S['summary']
    P(f'\n## E2, c_sw={csw} (exact on quantised chains unless marked MC)\n')
    P('| controller | J | hits/100 | switches/1000 | GUARD frac | frac of Bayes gain |\n|---|---|---|---|---|---|')
    Jf, JB = sm['J_bestfixed'], sm['J_Bayes']
    for k in ('C-W', 'C-G', 'C-Bayes', 'C-oracle'):
        v = S[k]
        P(f"| {k} | {v['J']:.6f} | {v['hits']:.4f} | {v['switches_per_1000']:.3f} | {v['guard_frac']:.4f} | {(v['J']-Jf)/(JB-Jf):.3f} |")
    if csw == '1.0':
        v = S['C-Bayes-myopic']
        P(f"| Bayes myopic (c_sw=0 policy) | {v['J']:.6f} | {v['hits']:.4f} | {v['switches_per_1000']:.3f} | {v['guard_frac']:.4f} | {(v['J']-Jf)/(JB-Jf):.3f} |")
    P('\n| lambda | best literal J | kind | theta_on | theta_off | hits | sw/1000 | frac | single J | single theta | matched single J | matched frac |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
    for lam in e2['lambdas']:
        r = S['sigma'][str(lam)]; b = r['literal_best']; s = r['literal_single']; m = r['matched_single']
        P(f"| {lam} | {b['J']:.6f} | {b['kind']} | {b['theta_on']:.4f} | {b['theta_off']:.4f} | {b['hits']:.4f} | {b['switches_per_1000']:.2f} | {b['frac_gain']:.3f} | {s['J']:.6f} | {s['theta']:.4f} | {m['J']:.6f} | {m['frac_gain']:.3f} |")
    if 'structure' in S['C-Bayes']:
        P('\nBayes belief-MDP policy structure: %s; RVI gain %.8f (span %.1e) vs chain evaluation %.8f' % (S['C-Bayes']['structure'], S['C-Bayes']['rvi_gain'], S['C-Bayes']['rvi_span'], S['C-Bayes']['J']))
    P('\nCross-checks exact vs MC (test seed 303, 400 chains x 50000 steps after burn-in; CRN):')
    for x in S['crosscheck']:
        P('- ' + json.dumps(x))
P('\nGrid convergence (exact J at fixed thresholds):')
for g in e2['grid_convergence']:
    P('- ' + json.dumps(g))
P('\n## E3 phase diagram (exact, N=5001; single-threshold sigma, literal timing)\n')
P('| G | r_g | k_G | best fixed | analytic window | oracle gain | Bayes gain | sigma gain | frac | best lambda |\n|---|---|---|---|---|---|---|---|---|---|')
for c in e3['phase']:
    fr = '%.3f' % c['frac_captured'] if c['frac_captured'] is not None else '-'
    P(f"| {c['G']} | {c['r_g']} | {c['k_G']:.2f} | {c['bestfixed']} | {c['analytic_condition_switching_useful']} | {c['oracle_gain']:.5f} | {max(c['Bayes_gain'],0):.5f} | {max(c['sigma_gain'],0):.5f} | {fr} | {c['sigma_best_lambda'] if c['Bayes_gain']>1e-9 else '-'} |")
P('\nPersistence sweep (G=0.8, r_g=0.7):\n\n| scale | q01 | q10 | 1/q01 | 1/q10 | Bayes gain | sigma gain | frac | best lambda | oracle gain |\n|---|---|---|---|---|---|---|---|---|---|')
for c in e3['persistence']:
    P(f"| {c['scale']} | {c['q01']:.5f} | {c['q10']:.4f} | {c['inv_q01']:.0f} | {c['inv_q10']:.2f} | {c['Bayes_gain']:.5f} | {c['sigma_gain']:.5f} | {c['frac_captured']:.3f} | {c['sigma_best_lambda']} | {c['oracle_gain']:.4f} |")
P('\n## E4 (MC, %d chains x %d steps per config, burn-in %d, CRN across configs; config %s)\n' % (e4['mc']['chains_per_config'], e4['mc']['T'], e4['mc']['burn'], e4['config']))
P('| rho | fuse | J (se) | GUARD frac (se) | J - J_nofuse paired (se) | dwell after storm: mean / p95 / max (censored) | max GUARD count after last real ann. | max span |\n|---|---|---|---|---|---|---|---|')
for r in e4['rows']:
    f = 'none' if r['fuse'] is None else '%d/%d' % (r['fuse']['T_max'], r['fuse']['P'])
    d = r['dwell_after_storm_end']
    P(f"| {r['rho']} | {f} | {r['J']:.5f} ({r['J_se']:.5f}) | {r['guard_frac']:.4f} ({r['guard_frac_se']:.4f}) | {r['J_minus_nofuse_paired']:+.6f} ({r['J_minus_nofuse_paired_se']:.6f}) | {d['mean']:.1f} / {d['p95']:.0f} / {d['max']:.0f} ({d['n_censored']}) | {r['max_guard_count_after_last_real_announcement']} | {r['max_guard_steps_after_last_real_announcement']} |")
P('\nExact (quantised N=5001) vs MC, no fuse:')
for x in e4['crosscheck_exact_vs_mc']:
    P('- ' + json.dumps(x))
P('\n## E5 (exact on belief grid NB=1001; c* curve from a 71-point c_r sweep with linear interpolation; bisection checks)\n')
P('Validation p0=p1=0.05: ' + json.dumps(e5['validation_p0_eq_p1']))
P('\n| b | C-W full | C-W shortcut | Bayes full | Bayes shortcut |\n|---|---|---|---|---|')
for b in e5['C-W']['at']:
    P(f"| {b} | {e5['C-W']['at'][b]['full']:.4f} | {e5['C-W']['at'][b]['shortcut']:.4f} | {e5['C-Bayes']['at'][b]['full']:.4f} | {e5['C-Bayes']['at'][b]['shortcut']:.4f} |")
for n in ('C-W', 'C-Bayes'):
    v = e5[n]
    P(f"\n{n}: stationary average full {v['stationary_avg_full']:.4f}, shortcut {v['stationary_avg_shortcut']:.4f}; full nonincreasing {v['full_nonincreasing']} (max step {v['full_max_increase']:.4f}); full-shortcut in [{v['full_minus_shortcut_min']:.2e}, {v['full_minus_shortcut_max']:.4f}]; bisection checks {json.dumps(v['bisect_checks'])}")
open(R + 'tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
