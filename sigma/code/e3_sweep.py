"""E3: when does cortisol matter. (a) G x r_g phase diagram at c_sw=0: best fixed mode, Bayes gain,
best single-threshold sigma gain (literal timing), all on quantised chains (N=5001; E2 shows the
quantisation error is ~1e-6 at N=5001-10001). The stationary laws do not depend on (G, r_g), so
they are computed once per filter. (b) Regime-persistence sweep: (q01,q10) scaled by s."""
import json, time
import numpy as np
from chain import *
from e2_mode import bayes_threshold_c0

t0 = time.time()
N = 5001
LAMS = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
GS = [1.0, 0.9, 0.8, 0.65, 0.5, 0.25, 0.0]
RGS = [0.5, 0.6, 0.7, 0.8, 0.9]
base = dict(REG)


def laws(prm):
    Pb = build_filter_chain(prm, bayes_update(prm), N, 'matched')
    L = {'bayes': (Pb, stationary(Pb))}
    for lam in LAMS:
        P = build_filter_chain(prm, sigma_update(lam), N, 'literal')
        L[lam] = (P, stationary(P))
    return L


def cell(prm, L):
    fm = fixed_mode_values(prm)
    Jfix = max(fm['J_W'], fm['J_G'])
    orc = oracle_value(prm)
    Pb, pib = L['bayes']
    famb = threshold_family(prm, Pb, pib, N)
    JB = float(famb['J0'].max())                       # best threshold on b (= myopic rule, c_sw=0)
    bstar = bayes_threshold_c0(prm)
    best = (-1, None, None)
    per_lam = {}
    for lam in LAMS:
        P, pi = L[lam]
        fam = threshold_family(prm, P, pi, N)
        k = int(np.argmax(fam['J0']))
        per_lam[str(lam)] = float(fam['J0'][k])
        if fam['J0'][k] > best[0]:
            best = (float(fam['J0'][k]), lam, float(fam['theta'][k]))
    k = kG(prm['G'], prm['h'])
    cond = bool(prm['p0'] * k < 1 - prm['r_g'] < prm['p1'] * k)
    gB = JB - Jfix
    return dict(G=prm['G'], r_g=prm['r_g'], k_G=k, J_W=fm['J_W'], J_G=fm['J_G'],
                bestfixed='W' if fm['J_W'] >= fm['J_G'] else 'G', J_bestfixed=Jfix,
                J_oracle=orc['J'], oracle_gain=orc['J'] - Jfix, J_Bayes=JB, Bayes_gain=gB, Bayes_b_star=bstar,
                J_sigma=best[0], sigma_gain=best[0] - Jfix, sigma_best_lambda=best[1], sigma_best_theta=best[2],
                frac_captured=(best[0] - Jfix) / gB if gB > 1e-9 else None,
                analytic_condition_switching_useful=cond, sigma_J_by_lambda=per_lam)


res = dict(N_grid=N, lambdas=LAMS, G_list=GS, rg_list=RGS)
L0 = laws(base)
print(f'laws {time.time()-t0:.1f}s', flush=True)
cells = []
for G in GS:
    for rg in RGS:
        c = cell(dict(base, G=G, r_g=rg), L0)
        cells.append(c)
        print(f"G={G} r_g={rg}: fixed={c['bestfixed']} cond={c['analytic_condition_switching_useful']} "
              f"Bayes gain={c['Bayes_gain']:.5f} sigma gain={c['sigma_gain']:.5f} lam*={c['sigma_best_lambda']} "
              f"oracle gain={c['oracle_gain']:.5f}", flush=True)
res['phase'] = cells
# consistency of the analytic condition: Bayes gain > 1e-9 implies condition, and vice versa?
res['phase_check'] = dict(
    cond_false_and_bayes_gain_positive=[(c['G'], c['r_g'], c['Bayes_gain']) for c in cells
                                        if not c['analytic_condition_switching_useful'] and c['Bayes_gain'] > 1e-9],
    cond_true_and_bayes_gain_zero=[(c['G'], c['r_g'], c['Bayes_gain']) for c in cells
                                   if c['analytic_condition_switching_useful'] and c['Bayes_gain'] <= 1e-9])
# persistence sweep at baseline G, r_g
pers = []
for s in [0.25, 0.5, 1, 2, 4, 8]:
    prm = dict(base, q01=base['q01'] * s, q10=base['q10'] * s)
    L = L0 if s == 1 else laws(prm)
    c = cell(prm, L)
    c.update(scale=s, q01=prm['q01'], q10=prm['q10'], inv_q01=1 / prm['q01'], inv_q10=1 / prm['q10'])
    pers.append(c)
    print(f"scale={s}: Bayes gain={c['Bayes_gain']:.5f} sigma gain={c['sigma_gain']:.5f} lam*={c['sigma_best_lambda']} "
          f"frac={c['frac_captured']} oracle gain={c['oracle_gain']:.5f} {time.time()-t0:.0f}s", flush=True)
res['persistence'] = pers
res['wall_s'] = time.time() - t0
json.dump(res, open('../results/e3.json', 'w'), indent=1, default=float)
print(f'E3 done {time.time()-t0:.1f}s')
