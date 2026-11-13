"""Targeted strict check: main-text claims of sec_sigma.tex against named result keys."""
import json
R='../results/'; L=lambda f: json.load(open(R+f))
e1,e2,e3,e4,e5,s0,dv=L('e1.json'),L('e2.json'),L('e3.json'),L('e4.json'),L('e5.json'),L('step0.json'),L('derived.json')
S0,S1=e2['by_csw']['0.0'],e2['by_csw']['1.0']
row=lambda r,f: [x for x in e4['rows'] if x['rho']==r and x['fuse']==f][0]
f100={'T_max':100,'P':20}
C=[('b*',S0['C-Bayes']['b_star'],0.6007,4),('JB',S0['C-Bayes']['J'],0.87369,5),('gain',dv['e2']['bayes_gain_c0'],0.01369,5),
('rel',dv['e2']['relative_gain_bayes_pct'],1.6,1),('of oracle',dv['e2']['bayes_over_oracle_gain_pct'],38.9,1),('oracle gain',dv['e2']['oracle_gain_c0'],0.0352,4),
('hitsB',S0['C-Bayes']['hits'],0.59,2),('guardB',S0['C-Bayes']['guard_frac']*100,14.5,1),('bestJ',S0['summary']['best_sigma_literal']['J'],0.87197,5),
('bestlam',S0['summary']['best_lambda_literal'],0.1,1),
('thon',S0['summary']['best_sigma_literal']['theta_on'],0.1530,4),('thoff',S0['summary']['best_sigma_literal']['theta_off'],0.1092,4),
('share',dv['e2']['frac_c0_pct'],87.4,1),('single005',S0['sigma']['0.05']['literal_single']['frac_gain']*100,86.5,1),('th005',S0['sigma']['0.05']['literal_single']['theta'],0.1116,4),
('matched',dv['e2']['matched_best_frac_pct'],93.1,1),('lam1',dv['e2']['frac_lam1_pct'],9.8,1),('lam0005',dv['e2']['frac_lam0005_pct'],3.2,1),
('l05s',dv['e2']['frac_lam05_single_pct'],27.8,1),('l05l',dv['e2']['frac_lam05_latch_pct'],66.5,1),
('bon',S1['C-Bayes']['structure']['b_on'],0.934,3),('boff',S1['C-Bayes']['structure']['b_off'],0.345,3),('JB1',S1['C-Bayes']['J'],0.86438,5),
('myloss',dv['e2']['myopic_bayes_loss_c1'],0.0075,4),('share1',dv['e2']['frac_c1_pct'],75.1,1),('sw1',S1['sigma']['0.1']['literal_best']['switches_per_1000'],6.3,1),
('miss',dv['e1']['0.1_0.2']['frac_missed_pct'],57.1,1),('hitsG1',dv['e1']['0.1_0.2']['hits_G1'],2.86,2),('miss012',dv['e1']['0.1_0.12']['frac_missed_pct'],27.5,1),
('nimp',e1['mc_setup']['n_impacts'],959195,0),
('calm',dv['e1_mc_pct']['0.1_0.2']['frac_missed_calm_pct'],95.7,1),('first',dv['e1_mc_pct']['0.1_0.2']['frac_missed_storm_first_pct'],98.2,1),('later',dv['e1_mc_pct']['0.1_0.2']['frac_missed_storm_later_pct'],31.9,1),
('window',dv['e3']['window_cells'],24,0),('gaincells',dv['e3']['gain_cells'],23,0),('thr065',dv['e3']['G065_rg09_threshold'],0.0241,4),
('frmin',dv['e3']['frac_range_pct'][0],14.7,1),('frmax',dv['e3']['frac_range_pct'][1],96.8,1),
('p025',e3['persistence'][0]['frac_captured']*100,95.1,1),('p8',e3['persistence'][5]['frac_captured']*100,2.1,1),('pg025',e3['persistence'][0]['Bayes_gain'],0.02517,5),('pg8',e3['persistence'][5]['Bayes_gain'],0.00034,5),
('plam025',e3['persistence'][0]['sigma_best_lambda'],0.03,2),('plam8',e3['persistence'][5]['sigma_best_lambda'],0.5,1),
('mf',dv['e4']['meanfield_calm'],0.0911,4),('g0',row(0.0,None)['guard_frac'],0.143,3),('g01',row(0.1,None)['guard_frac'],0.206,3),('g02',row(0.2,None)['guard_frac'],0.337,3),('g04',row(0.4,None)['guard_frac'],0.944,3),('g03',row(0.3,None)['guard_frac'],0.655,3),
('d0',row(0.0,None)['dwell_after_storm_end']['mean'],4.1,1),('d01',row(0.1,None)['dwell_after_storm_end']['mean'],12.6,1),('d02',row(0.2,None)['dwell_after_storm_end']['mean'],43.9,1),('d03',row(0.3,None)['dwell_after_storm_end']['mean'],232.5,1),('d04',row(0.4,None)['dwell_after_storm_end']['mean'],2239,0),
('std',dv['e4']['sigma_std_rho02'],0.094,3),('maxz',dv['e4']['max_abs_z'],0.72,2),('nf',dv['e4']['n_fused'],84,0),('nv',dv['e4']['n_naive_span_violations'],24,0),('mx',dv['e4']['max_naive_excess'],9,0),
('fc100',dv['e4']['fuse_cost_c0']['100_20']*1e5,1.9,1),('fcp',dv['e4']['fuse_cost_pct_of_gain']['100_20'],0.16,2),('fc50',dv['e4']['fuse_cost_c0']['50_20']*1e4,4.0,1),('fc50p',dv['e4']['fuse_cost_pct_of_gain']['50_20'],3.4,1),
('fc200',dv['e4']['fuse_cost_c0']['200_20'],0.0,6),('maxg',dv['e4']['max_guard_fuse100_20'],0.39,2),('maxg3',dv['e4']['max_guard_fuse100_20'],0.388,3),('g05',row(0.5,None)['guard_frac'],0.998,3),
('bounds',int(dv['e4']['all_count_and_span_bounds_hold']),1,0),('sej',dv['e4']['max_J_se'],0.0009,4),('seg',dv['e4']['max_guard_frac_se'],0.0036,4),
('cw0',e5['C-W']['at']['0.0']['full'],13.759,3),('cw1',e5['C-W']['at']['1.0']['full'],10.109,3),('cb0',e5['C-Bayes']['at']['0.0']['full'],13.777,3),('cb1',e5['C-Bayes']['at']['1.0']['full'],10.737,3),
('crossW',e5['C-W']['b_cross_cr_12_4787_full'][1],0.373,3),('crossB',e5['C-Bayes']['b_cross_cr_12_4787_full'][1],0.433,3),
('gapmax',max(e5['C-W']['full_minus_shortcut_max'],e5['C-Bayes']['full_minus_shortcut_max']),0.28,2),
('avgW',e5['C-W']['stationary_avg_full'],13.042,3),('avgB',e5['C-Bayes']['stationary_avg_full'],13.125,3),('avgd',dv['e5']['avg_diff_full'],0.083,3),
('shW',e5['C-W']['stationary_avg_shortcut'],12.974,3),('shWd',dv['e5']['cw_shortcut_avg_minus_prop']*1e6,6,0),('b061B',e5['C-Bayes']['at']['0.61']['full'],11.458,3),('b061W',e5['C-W']['at']['0.61']['full'],11.607,3),
('nonincW',int(e5['C-W']['full_nonincreasing']),1,0),('nonincB',int(e5['C-Bayes']['full_nonincreasing']),1,0),
('bis',dv['e5']['max_bisect_vs_sweep']*1e4,9.4,1),('val',dv['e5']['validation_max_err']*1e10,1.2,1),('paired',dv['e2']['max_abs_paired_z'],1.9,1),
('grid<2e-6',int(dv['e2']['grid_max_change_vs_10001']<2e-6),1,0),
('s0V',s0['C0 planner alone']['V_start'],8.2119,4),('s0c',s0['C0 planner alone']['cstar_bisect'],8.3591,4),('s3V',s0['C3 planner + zero-latency reflex']['V_start'],14.4440,4),
('s3c',s0['C3 planner + zero-latency reflex']['cstar_bisect'],14.1272,4),('s1c',s0['C1 one in-time tier G=0.8 phi=0.02']['cstar_bisect'],12.6910,4),
('s0agree',int(max(abs(v['bisect_minus_closed']) for k,v in s0.items() if isinstance(v,dict) and 'bisect_minus_closed' in v)<1e-10),1,0)]
bad=0
for n,v,t,d in C:
    if abs(round(float(v),d)-t)>1e-9: bad+=1; print('MISMATCH',n,v,t)
print(len(C),'checked,',bad,'mismatches')
json.dump(dict(n=len(C),mismatches=bad),open(R+'check_keys.json','w'))
