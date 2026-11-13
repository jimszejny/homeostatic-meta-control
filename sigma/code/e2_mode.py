"""E2: cortisol as a tonic mode setter. Controllers C-W, C-G, C-sigma (literal and info-matched
timing; single threshold exact on the quantised chain, two-threshold searched by MC then
evaluated exactly), C-Bayes (c_sw=0: myopic threshold on the predictive belief; c_sw=1: belief-MDP
by relative value iteration), C-oracle. Cross-checks quantised-chain vs Monte Carlo."""
import json, time, sys
import numpy as np
from chain import *
from chain import _interp

def bayes_paths(prm, a):
    """b_t = P(z_t=1|a_0..a_t) along axis 1, prior at t=0: stationary law."""
    f = bayes_update(prm)
    C, T = a.shape
    b = np.empty((C, T))
    pz = stationary_z(prm['q01'], prm['q10'])
    prior = np.full(C, pz[1])
    l1 = np.where(a[:, 0] == 1, prm['p1'], 1 - prm['p1']); l0 = np.where(a[:, 0] == 1, prm['p0'], 1 - prm['p0'])
    b[:, 0] = prior * l1 / (prior * l1 + (1 - prior) * l0)
    for t in range(1, T):
        b[:, t] = f(b[:, t - 1], a[:, t])
    return b


def bayes_threshold_c0(prm):
    """GUARD iff 1 - k_G * pi(b) < r_g, pi(b) = p0 + (p1-p0) P(z_{t+2}=1|b): threshold b*."""
    Q2 = np.linalg.matrix_power(Qmat(prm['q01'], prm['q10']), 2)
    pistar = (1 - prm['r_g']) / kG(prm['G'], prm['h'])
    s = (pistar - prm['p0']) / (prm['p1'] - prm['p0'])        # needed P(z_{t+2}=1|b)
    bstar = (s - Q2[0, 1]) / (Q2[1, 1] - Q2[0, 1])
    return float(bstar)


def bayes_rvi(prm, N=2001, tol=1e-11, maxit=200000):
    """Average-reward belief MDP with switch cost, state (b on grid, previously committed mode),
    action = mode committed now. Reward booked at commit time (the step it governs is 3 steps
    later; a constant shift does not change average reward). Relative value iteration with
    aperiodicity damping 0.5."""
    g = np.linspace(0, 1, N)
    f = bayes_update(prm)
    bm = g * (1 - prm['q10']) + (1 - g) * prm['q01']
    pa1 = prm['p0'] + (prm['p1'] - prm['p0']) * bm
    i1, w1 = _interp(f(g, np.ones(N)), N)
    i0, w0 = _interp(f(g, np.zeros(N)), N)
    def EV(h):
        return pa1 * ((1 - w1) * h[i1] + w1 * h[i1 + 1]) + (1 - pa1) * ((1 - w0) * h[i0] + w0 * h[i0 + 1])
    pim = bayes_impact_prob(prm, g)
    R = np.stack([1 - pim * kG(prm['G'], prm['h']), np.full(N, prm['r_g'])])   # [mode, b]
    h = np.zeros((2, N))        # h[prev_mode, b]
    for it in range(maxit):
        cont = np.stack([EV(h[0]), EV(h[1])])        # value of next state with prev=m
        Qv = np.empty((2, 2, N))                     # [prev, m, b]
        for pr in (0, 1):
            for m in (0, 1):
                Qv[pr, m] = R[m] - prm['c_sw'] * (m != pr) + cont[m]
        Th = Qv.max(1)
        gain_lo, gain_hi = (Th - h).min(), (Th - h).max()
        hn = 0.5 * h + 0.5 * (Th - Th[0, 0])
        if gain_hi - gain_lo < tol:
            break
        h = hn
    pol = Qv.argmax(1)          # [prev, b]
    return dict(gain=0.5 * (gain_lo + gain_hi), span=gain_hi - gain_lo, iters=it, pol=pol, grid=g)


def mc_hyst_search(prm, s, z, a, burn, th_on_list, ratios):
    best = None; table = []
    for th in th_on_list:
        for r in ratios:
            l = latch_path(s, th, r * th)
            o = mc_eval_modes(prm, l, z, a, burn)
            table.append((float(th), float(r * th), o['J']))
            if best is None or o['J'] > best[2]:
                best = (float(th), float(r * th), o['J'])
    return best, table


def main():
    t0 = time.time()
    N = 10001
    LAMS = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
    out = dict(N_grid=N, lambdas=LAMS, by_csw={})
    # MC data: training (selection) and test (evaluation / cross-check), independent seeds
    base = dict(REG)
    Ctr, Ttr, burn = 60, 23000, 3000
    ztr, atr, _ = simulate_regime(base, Ctr, Ttr, seed=202)
    Cte, Tte = 400, 53000
    zte, ate, _ = simulate_regime(base, Cte, Tte, seed=303)
    bte = bayes_paths(base, ate)
    out['mc_setup'] = dict(train=dict(C=Ctr, T=Ttr, burn=burn, seed=202), test=dict(C=Cte, T=Tte, burn=burn, seed=303))
    # stationary laws per lambda (independent of c_sw)
    chains = {}
    for lam in LAMS:
        for timing in ('literal', 'matched'):
            P = build_filter_chain(base, sigma_update(lam), N, timing)
            chains[(lam, timing)] = (P, stationary(P))
    Pb = build_filter_chain(base, bayes_update(base), N, 'matched')
    pib = stationary(Pb)
    print(f'chains built {time.time()-t0:.1f}s', flush=True)

    for csw in (0.0, 1.0):
        prm = dict(base, c_sw=csw)
        R = {}
        fm = fixed_mode_values(prm)
        R['C-W'] = dict(J=fm['J_W'], hits=fm['hits_W'], switches_per_1000=0.0, guard_frac=0.0, method='exact closed form')
        R['C-G'] = dict(J=fm['J_G'], hits=0.0, switches_per_1000=0.0, guard_frac=1.0, method='exact closed form')
        R['C-oracle'] = dict(oracle_value(prm), method='exact closed form (naive oracle pays c_sw per regime change)')
        Jfix = max(fm['J_W'], fm['J_G'])
        # ---- Bayes
        bstar = bayes_threshold_c0(prm)
        famb = threshold_family(prm, Pb, pib, N)
        Jb = famb['J0'] - csw * famb['sw']
        kstar = int(np.searchsorted(np.linspace(0, 1, N), bstar))
        R['C-Bayes-myopic'] = dict(b_star=bstar, J=float(Jb[kstar]), J_noswitchcost=float(famb['J0'][kstar]),
                                   hits=float(famb['hits'][kstar]), switches_per_1000=1000 * float(famb['sw'][kstar]),
                                   guard_frac=float(famb['guard'][kstar]),
                                   best_grid_threshold=float(famb['theta'][np.argmax(Jb)]), J_best_grid_threshold=float(Jb.max()),
                                   method='quantised (b,z) chain, N=%d, exact stationary law' % N)
        if csw > 0:
            rv = bayes_rvi(prm)
            pol = rv['pol']; g2 = rv['grid']
            # structure: for prev=W, GUARD set; for prev=G, GUARD set
            def runs(x):
                return int(np.sum(np.abs(np.diff(x.astype(int)))))
            b_on = float(g2[np.argmax(pol[0] == 1)]) if pol[0].any() else None
            b_off = float(g2[np.argmax(pol[1] == 1)]) if pol[1].any() else None
            struct = dict(b_on=b_on, b_off=b_off, n_changes_prevW=runs(pol[0]), n_changes_prevG=runs(pol[1]),
                          two_threshold=bool(runs(pol[0]) == 1 and runs(pol[1]) == 1 and b_off < b_on))
            Pl, D, Z, LP = build_latch_chain(prm, bayes_update(prm), N, b_on, b_off, 'matched')
            ev = eval_latch(prm, stationary(Pl), D, Z, LP)
            lb = latch_path(bte, b_on, b_off)
            mc = mc_eval_modes(prm, lb, zte, ate, burn)
            R['C-Bayes'] = dict(ev, rvi_gain=rv['gain'], rvi_span=rv['span'], rvi_iters=rv['iters'], structure=struct,
                                mc=mc, method='belief-MDP RVI (N=2001, average reward) -> policy evaluated on quantised (b,l,z) chain N=%d' % N)
        else:
            lb = (bte >= bstar).astype(np.int8)
            R['C-Bayes'] = dict(R['C-Bayes-myopic'], mc=mc_eval_modes(prm, lb, zte, ate, burn))
        JB = R['C-Bayes']['J']
        # ---- sigma
        sig = {}
        for lam in LAMS:
            row = {}
            for timing in ('literal', 'matched'):
                P, pi = chains[(lam, timing)]
                fam = threshold_family(prm, P, pi, N)
                Js = fam['J0'] - csw * fam['sw']
                k = int(np.argmax(Js))
                row[timing + '_single'] = dict(theta=float(fam['theta'][k]), J=float(Js[k]), J_noswitchcost=float(fam['J0'][k]),
                                               hits=float(fam['hits'][k]), switches_per_1000=1000 * float(fam['sw'][k]),
                                               guard_frac=float(fam['guard'][k]))
            # two-threshold (literal): MC search on training data, exact evaluation
            # coarse absolute grid, then a local refinement around the coarse optimum
            lit_tr, _ = sigma_paths(atr, lam)
            coarse = np.geomspace(0.005, 1.0, 20)
            ratios = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15, 0.1, 0.05, 0.02]
            bst, _ = mc_hyst_search(prm, lit_tr, ztr, atr, burn, coarse, ratios)
            r0 = bst[1] / bst[0]
            fine_on = np.clip(bst[0] * np.geomspace(0.8, 1.25, 7), 1e-4, 1.0)
            fine_r = np.clip(r0 * np.geomspace(0.6, 1.0 / max(r0, 0.6), 7), 0.005, 1.0)
            bst2, _ = mc_hyst_search(prm, lit_tr, ztr, atr, burn, np.unique(fine_on), np.unique(fine_r))
            if bst2[2] > bst[2]:
                bst = bst2
            del lit_tr
            Pl, D, Z, LP = build_latch_chain(prm, sigma_update(lam), N, bst[0], bst[1], 'literal')
            ev = eval_latch(prm, stationary(Pl), D, Z, LP)
            row['literal_two'] = dict(ev, theta_on=bst[0], theta_off=bst[1], J_train_mc=bst[2])
            # overall literal best
            if row['literal_two']['J'] > row['literal_single']['J']:
                row['literal_best'] = dict(row['literal_two'], kind='two-threshold')
            else:
                row['literal_best'] = dict(row['literal_single'], theta_on=row['literal_single']['theta'],
                                           theta_off=row['literal_single']['theta'], kind='single')
            for key in ('literal_best', 'matched_single', 'literal_single'):
                row[key]['frac_gain'] = (row[key]['J'] - Jfix) / (JB - Jfix)
            sig[str(lam)] = row
            print(f"csw={csw} lam={lam}: single J={row['literal_single']['J']:.6f} th={row['literal_single']['theta']:.4f} | "
                  f"two J={row['literal_two']['J']:.6f} ({bst[0]:.4f},{bst[1]:.4f}) | matched J={row['matched_single']['J']:.6f} | "
                  f"frac={row['literal_best']['frac_gain']:.3f}  {time.time()-t0:.0f}s", flush=True)
        best_lam = max(LAMS, key=lambda l: sig[str(l)]['literal_best']['J'])
        best_lam_m = max(LAMS, key=lambda l: sig[str(l)]['matched_single']['J'])
        R['sigma'] = sig
        R['summary'] = dict(J_bestfixed=Jfix, bestfixed='C-W' if fm['J_W'] >= fm['J_G'] else 'C-G', J_Bayes=JB,
                            J_oracle=R['C-oracle']['J'], best_lambda_literal=best_lam,
                            best_sigma_literal=sig[str(best_lam)]['literal_best'],
                            best_lambda_matched=best_lam_m, best_sigma_matched=sig[str(best_lam_m)]['matched_single'],
                            regime_timescales=dict(inv_q01=1 / prm['q01'], inv_q10=1 / prm['q10'],
                                                   inv_q01_plus_q10=1 / (prm['q01'] + prm['q10'])))
        # ---- MC cross-checks on the test data (independent seed) for three sigma configs
        xc = []
        mcW = mc_eval_modes(prm, np.zeros_like(ate), zte, ate, burn)
        jw_ex = fm['J_W']
        def dz(mc, ex):   # CRN: compare (J_ctrl - J_W) between MC (paired, per chain) and exact
            d = mc['J_chain'] - mcW['J_chain']
            return dict(diff=float(d.mean() - (ex - jw_ex)), paired_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                        z=float((d.mean() - (ex - jw_ex)) / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-300)))
        xc.append(dict(name='C-W', exact=jw_ex, mc=mcW['J'], se=mcW['J_se'], z=(mcW['J'] - jw_ex) / mcW['J_se']))
        for lam in (best_lam, 0.02, 1):
            bl = sig[str(lam)]['literal_best']
            lit, mat = sigma_paths(ate, lam)
            l = latch_path(lit, bl['theta_on'], bl['theta_off'])
            mc = mc_eval_modes(prm, l, zte, ate, burn)
            ms = sig[str(lam)]['matched_single']
            mcm = mc_eval_modes(prm, (mat >= ms['theta'] - 1e-12).astype(np.int8), zte, ate, burn)
            xc.append(dict(lam=lam, literal=dict(exact=bl['J'], mc=mc['J'], se=mc['J_se'], mc_raw=mc['J_raw'], se_raw=mc['J_raw_se'],
                                                 z=(mc['J'] - bl['J']) / mc['J_se'], diff_vs_W_mc_minus_exact=dz(mc, bl['J']), hits_exact=bl['hits'], hits_mc=mc['hits'],
                                                 sw_exact=bl['switches_per_1000'], sw_mc=mc['switches_per_1000']),
                           matched=dict(exact=ms['J'], mc=mcm['J'], se=mcm['J_se'], z=(mcm['J'] - ms['J']) / mcm['J_se'], diff_vs_W_mc_minus_exact=dz(mcm, ms['J']))))
            del lit, mat
        xc.append(dict(name='Bayes', exact=JB, mc=R['C-Bayes']['mc']['J'], se=R['C-Bayes']['mc']['J_se'],
                       z=(R['C-Bayes']['mc']['J'] - JB) / R['C-Bayes']['mc']['J_se'], diff_vs_W_mc_minus_exact=dz(R['C-Bayes']['mc'], JB)))
        R['crosscheck'] = xc
        for x in xc:
            print('xcheck', strip_chain(x), flush=True)
        out['by_csw'][str(csw)] = R
        print(f"csw={csw}: Jfix={Jfix:.6f} JB={JB:.6f} Joracle={R['C-oracle']['J']:.6f} best lam={best_lam}", flush=True)
    # quantisation check: same controllers on coarser/finer grids
    gc = []
    prm = dict(base)
    for Nx in (2501, 5001, 10001, 20001):
        row = dict(N=Nx)
        P = build_filter_chain(prm, bayes_update(prm), Nx, 'matched'); pi = stationary(P)
        fam = threshold_family(prm, P, pi, Nx); k = int(np.searchsorted(np.linspace(0, 1, Nx), bayes_threshold_c0(prm)))
        row['Bayes_c0'] = float(fam['J0'][k])
        for lam, th in ((0.1, 0.1263), (0.02, 0.1145), (0.005, 0.1214)):
            P = build_filter_chain(prm, sigma_update(lam), Nx, 'literal'); pi = stationary(P)
            fam = threshold_family(prm, P, pi, Nx); k = int(np.searchsorted(np.linspace(0, 1, Nx), th - 1e-12))
            row['sigma_%g_theta_%g' % (lam, th)] = float(fam['J0'][k])
        gc.append(row); print('grid', row, flush=True)
    out['grid_convergence'] = gc
    out['wall_s'] = time.time() - t0
    json.dump(strip_chain(out), open('../results/e2.json', 'w'), indent=1, default=float)
    print(f'E2 done {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
