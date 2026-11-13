"""(revision fix1: p95 rounded with %.0f instead of truncated by %d; E4/E5 captions extended.)
Fill app_template.tex with tables from results/*.json -> paper/app_sigma.tex."""
import json, re, os
R = '../results/'
OUT = '../../paper/app_sigma.tex'
e1 = json.load(open(R + 'e1.json')); e2 = json.load(open(R + 'e2.json')); e3 = json.load(open(R + 'e3.json'))
e4 = json.load(open(R + 'e4.json')); e5 = json.load(open(R + 'e5.json'))
s = open('app_template.tex').read()


def tex_escape(t):
    return (t.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
             .replace('#', r'\#').replace('<=', r'$\le$').replace('>=', r'$\ge$').replace('<', r'$<$').replace('>', r'$>$'))


def f(x, d):
    return ('%.' + str(d) + 'f') % x


S0 = e2['by_csw']['0.0']
s = s.replace('%%BSTAR%%', f(S0['C-Bayes']['b_star'], 4))
pm = open(R + 'predictions.md').read()
ts = re.search(r'Written: (\S+)', pm).group(1)
s = s.replace('%%TIMESTAMP%%', ts.replace('T', ' ').replace('Z', ''))
s = s.replace('%%SHA%%', open(R + 'predictions.sha256').read().split()[0])
items = []
for line in pm.splitlines():
    m = re.match(r'^(Q\d|M\d)(?: \(mine\))?\. (.*)$', line)
    if m:
        lab = m.group(1) + (' (mine)' if '(mine)' in line.split('.')[0] else '')
        items.append(r'\item[%s.] %s' % (lab, tex_escape(m.group(2))))
s = s.replace('%%PREDS%%', '\n'.join(items))

# cross-check table
L = [r'\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}llrrrr@{}}\toprule',
     r'experiment & configuration & exact & Monte Carlo (s.e.) & paired diff. & $z$\\\midrule']
for csw in ('0.0', '1.0'):
    for x in e2['by_csw'][csw]['crosscheck']:
        if 'name' in x:
            if x['name'] == 'C-W':
                L.append(r'E2, $c_{\mathrm{sw}}=%g$ & always \WORK & %s & %s (%s) & -- & %.2f\\' % (float(csw), f(x['exact'], 5), f(x['mc'], 5), f(x['se'], 5), x['z']))
            else:
                dd = x['diff_vs_W_mc_minus_exact']
                L.append(r'E2, $c_{\mathrm{sw}}=%g$ & Bayes & %s & %s (%s) & %s & %.2f\\' % (float(csw), f(x['exact'], 5), f(x['mc'], 5), f(x['se'], 5), '%.1e' % dd['diff'], dd['z']))
        else:
            lt = x['literal']; dd = lt['diff_vs_W_mc_minus_exact']
            if dd['paired_se'] == 0:
                continue      # identical to always-WORK (no GUARD steps): nothing to check
            L.append(r'E2, $c_{\mathrm{sw}}=%g$ & $\sigma$ latch, $\lambda=%g$ & %s & %s (%s) & %s & %.2f\\' % (float(csw), x['lam'], f(lt['exact'], 5), f(lt['mc'], 5), f(lt['se'], 5), '%.1e' % dd['diff'], dd['z']))
for x in e1['crosscheck']:
    L.append(r'E1 & missed, $\lambda=%g$, $\theta=%g$ & %s & %s (%s) & -- & %.2f\\' % (x['lam'], x['theta'], f(x['exact'], 4), f(x['mc'], 4), f(x['mc_se'], 4), x['z']))
for x in e4['crosscheck_exact_vs_mc']:
    L.append(r'E4 & no fuse, $\psi=%g$: $J$ & %s & %s (%s) & -- & %.2f\\' % (x['rho'], f(x['exact_J'], 5), f(x['mc_J'], 5), f(x['mc_J_se'], 5), x['z_J']))
    L.append(r'E4 & no fuse, $\psi=%g$: \GUARD\ frac. & %s & %s (%s) & -- & %.2f\\' % (x['rho'], f(x['exact_guard'], 4), f(x['mc_guard'], 4), f(x['mc_guard_se'], 4), x['z_guard']))
L += [r'\bottomrule\end{tabular}',
      r'\caption{Exact (quantised chain) against Monte Carlo. E2: independent stream (seed 303), 400 chains of 50{,}000 steps after a 3000-step burn-in, common random numbers; ``paired diff.\textquotedblright{} is $(J_{\mathrm{MC}}-J^{\mathrm{MC}}_{\WORK})-(J_{\mathrm{exact}}-0.86)$ with its $z$ from the per-chain paired standard error. The unpaired $z\approx2$ of every E2 row is one shared fluctuation of the sampled storm fraction (always-\WORK\ itself is $2.0$ s.e.\ high). E1: seed 101, 400 chains of 48{,}000 steps. E4: 64 chains of 56{,}000 steps, quantised chain with $N=5001$.}',
      r'\label{tab:sigma-app-checks}\end{table}', '',
      r'Quantisation (E2, exact $J$ at fixed thresholds, $N=2501,5001,10001,20001$): ' +
      '; '.join(r'%s: %s' % (lab, ', '.join(f(g[k], 7) for g in e2['grid_convergence']))
                for k, lab in (('Bayes_c0', 'Bayes'), ('sigma_0.1_theta_0.1263', r'$\sigma$ ($0.1$, $0.1263$)'),
                               ('sigma_0.02_theta_0.1145', r'$\sigma$ ($0.02$, $0.1145$)'), ('sigma_0.005_theta_0.1214', r'$\sigma$ ($0.005$, $0.1214$)'))) + '.']
s = s.replace('%%TAB_CHECKS%%', '\n'.join(L))

# E1 table
ths = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
L = [r'\begin{table}[h]\centering\small', r'\begin{tabular}{@{}r' + 'r' * len(ths) + '@{}}\\toprule',
     r'$\lambda$ & ' + ' & '.join(r'$\theta=%g$' % t for t in ths) + r'\\\midrule']
for lam in e1['lambdas']:
    rows = {round(r['theta'], 4): r for r in e1['exact'][str(lam)]}
    L.append('%g & ' % lam + ' & '.join(f(rows[round(t, 4)]['frac_missed'], 4) for t in ths) + r'\\')
L += [r'\bottomrule\end{tabular}', r'\caption{E1 (exact, $N=10001$): fraction of impacts at which the filtered gate $\sigma_t\ge\theta$ is off. Hits per 100 steps of an always-\WORK\ planner are $5\Gamma m+5(1-\Gamma)$; the raw gate has $m=0$.}', r'\label{tab:sigma-app-e1}\end{table}']
s = s.replace('%%TAB_E1%%', '\n'.join(L))

# E2 table
L = [r'\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{@{}r|rrrrr|rrr|r|rrrr@{}}\toprule',
     r'& \multicolumn{5}{c|}{$c_{\mathrm{sw}}=0$, best latch} & \multicolumn{3}{c|}{single threshold} & matched & \multicolumn{4}{c}{$c_{\mathrm{sw}}=1$, best latch}\\',
     r'$\lambda$ & $J$ & $\theta_{\mathrm{on}}$ & $\theta_{\mathrm{off}}$ & hits & share & $J$ & $\theta$ & share & share & $J$ & $\theta_{\mathrm{on}}$ & $\theta_{\mathrm{off}}$ & share\\\midrule']
S1 = e2['by_csw']['1.0']
for lam in e2['lambdas']:
    a = S0['sigma'][str(lam)]; b = S1['sigma'][str(lam)]['literal_best']
    lb, ls, ms = a['literal_best'], a['literal_single'], a['matched_single']
    th1 = (r'$\infty$', r'$\infty$') if b['theta_on'] == float('inf') else (f(b['theta_on'], 4), f(b['theta_off'], 4))
    L.append('%g & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s\\\\' % (
        lam, f(lb['J'], 5), f(lb['theta_on'], 4), f(lb['theta_off'], 4), f(lb['hits'], 3), f(lb['frac_gain'], 3),
        f(ls['J'], 5), f(ls['theta'], 4), f(ls['frac_gain'], 3), f(ms['frac_gain'], 3),
        f(b['J'], 5), th1[0], th1[1], f(max(b['frac_gain'], 0.0), 3)))
L += [r'\bottomrule\end{tabular}',
      r'\caption{E2 for every $\lambda$ (exact on quantised chains, $N=10001$). ``Share\textquotedblright{}: of the Bayes gain at the same $c_{\mathrm{sw}}$ (Bayes $J=%s$ at $c_{\mathrm{sw}}=0$, $%s$ at $c_{\mathrm{sw}}=1$). A latch with $\theta_{\mathrm{on}}=\theta_{\mathrm{off}}$ is a single threshold; $\infty$: never \GUARD. Latches were selected by a Monte-Carlo search (seed 202, 60 chains of 20{,}000 steps; a 20-point log grid of $\theta_{\mathrm{on}}\in[0.005,1]$ times 10 ratios $\theta_{\mathrm{off}}/\theta_{\mathrm{on}}\in[0.02,1]$, then a $7\times7$ local refinement) and evaluated exactly; the search may miss the optimum slightly.}' % (f(S0['summary']['J_Bayes'], 5), f(S1['summary']['J_Bayes'], 5)),
      r'\label{tab:sigma-app-e2}\end{table}']
s = s.replace('%%TAB_E2%%', '\n'.join(L))

# E3 table: rows Gamma, columns r_g, entries Bayes gain / share
RG = e3['rg_list']
L = [r'\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}r' + 'c' * len(RG) + '@{}}\\toprule',
     r'$\Gamma$ & ' + ' & '.join(r'$r_{\mathrm g}=%g$' % r for r in RG) + r'\\\midrule']
for G in e3['G_list']:
    cells = []
    for rg in RG:
        c = [x for x in e3['phase'] if x['G'] == G and x['r_g'] == rg][0]
        w = '' if c['analytic_condition_switching_useful'] else r'$^\dagger$'
        if c['Bayes_gain'] > 1e-9:
            cells.append(r'%s / %s (%g)%s' % (f(c['Bayes_gain'], 5), f(c['frac_captured'], 3), c['sigma_best_lambda'], w))
        else:
            cells.append(r'fixed %s%s' % (r'\WORK' if c['bestfixed'] == 'W' else r'\GUARD', w))
    L.append('%g & ' % G + ' & '.join(cells) + r'\\')
L += [r'\bottomrule\end{tabular}', r'\caption{E3 (exact, $N=5001$, $c_{\mathrm{sw}}=0$): Bayes gain over the best fixed mode / share captured by the best single-threshold $\sigma$ (its $\lambda$). $^\dagger$: outside the analytic window.}', r'\label{tab:sigma-app-e3}\end{table}']
s = s.replace('%%TAB_E3%%', '\n'.join(L))

L = [r'\begin{table}[h]\centering\small', r'\begin{tabular}{@{}rrrrrrrr@{}}\toprule',
     r'scale & $q_{01}$ & $q_{10}$ & oracle gain & Bayes gain & best-$\sigma$ gain & share & best $\lambda$\\\midrule']
for c in e3['persistence']:
    L.append(r'%g & %s & %s & %s & %s & %s & %s & %g\\' % (c['scale'], f(c['q01'], 5), f(c['q10'], 4), f(c['oracle_gain'], 4), f(c['Bayes_gain'], 5), f(c['sigma_gain'], 5), f(c['frac_captured'], 3), c['sigma_best_lambda']))
L += [r'\bottomrule\end{tabular}', r'\caption{E3, regime persistence ($\Gamma=0.8$, $r_{\mathrm g}=0.7$, $c_{\mathrm{sw}}=0$; exact, $N=5001$; single-threshold $\sigma$).}', r'\label{tab:sigma-app-pers}\end{table}']
s = s.replace('%%TAB_PERS%%', '\n'.join(L))

# E4 full
fz = [None, {'T_max': 50, 'P': 20}, {'T_max': 100, 'P': 20}, {'T_max': 200, 'P': 20}, {'T_max': 100, 'P': 10}]
def gr(r, z):
    return [x for x in e4['rows'] if x['rho'] == r and x['fuse'] == z][0]
L = [r'\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3.2pt}', r'\begin{tabular}{@{}r|rrrrr|rrrrr|rr@{}}\toprule',
     r'& \multicolumn{5}{c|}{\GUARD\ fraction} & \multicolumn{5}{c|}{$J$} & \multicolumn{2}{c}{no fuse: dwell}\\',
     r'$\psi$ & none & 50/20 & 100/20 & 200/20 & 100/10 & none & 50/20 & 100/20 & 200/20 & 100/10 & mean & p95\\\midrule']
for r in sorted(set(x['rho'] for x in e4['rows'])):
    d = gr(r, None)['dwell_after_storm_end']
    L.append('%g & ' % r + ' & '.join(f(gr(r, z)['guard_frac'], 3) for z in fz) + ' & ' + ' & '.join(f(gr(r, z)['J'], 4) for z in fz) + ' & %s & %.0f\\\\' % (f(d['mean'], 1), d['p95']))
L += [r'\bottomrule\end{tabular}', r'\caption{E4 (Monte Carlo, seed 404; 64 chains of 56{,}000 steps per cell, common random numbers across cells; s.e.\ $\le0.0036$ for fractions, $\le0.0009$ for $J$). Fuse columns: $T_{\mathrm{dw}}/T_{\mathrm{pr}}$. Dwell: committed \GUARD\ steps from the first calm step after a storm until the first committed \WORK, censored at the end of the run, so mean and p95 are lower bounds (censored measurements: 2 at $\psi=0$, 28 of 7562 at $\psi=0.3$, 63 of 1440 at $\psi=0.4$, 64 of 128 at $\psi=0.5$). Of the six fuse settings run ($T_{\mathrm{dw}}\in\{50,100,200\}$, $T_{\mathrm{pr}}\in\{10,20\}$; 84 fused cells), four are shown. The count bound $T_{\mathrm{dw}}$ and the span bound $T_{\mathrm{dw}}+T_{\mathrm{pr}}-1$ of Proposition~\ref{prop:sigma-fuse} held in all 84 fused cells; the span bound $T_{\mathrm{dw}}$ failed in 24.}',
      r'\label{tab:sigma-app-e4}\end{table}']
s = s.replace('%%TAB_E4%%', '\n'.join(L))

L = [r'\begin{table}[h]\centering\small', r'\begin{tabular}{@{}rrrrr@{}}\toprule',
     r'$b$ & always \WORK: $\crs$ & $c_{\mathrm{ind}}$ & Bayes: $\crs$ & $c_{\mathrm{ind}}$\\\midrule']
for b in e5['C-W']['at']:
    L.append(r'%s & %s & %s & %s & %s\\' % (b, f(e5['C-W']['at'][b]['full'], 4), f(e5['C-W']['at'][b]['shortcut'], 4), f(e5['C-Bayes']['at'][b]['full'], 4), f(e5['C-Bayes']['at'][b]['shortcut'], 4)))
L.append(r'\midrule stationary mean & %s & %s & %s & %s\\' % (f(e5['C-W']['stationary_avg_full'], 4), f(e5['C-W']['stationary_avg_shortcut'], 4), f(e5['C-Bayes']['stationary_avg_full'], 4), f(e5['C-Bayes']['stationary_avg_shortcut'], 4)))
chk = '; '.join(r'%s, $b=%g$: bisection %s, sweep %s' % (n.replace('C-W', r'always \WORK').replace('C-Bayes', 'Bayes'), c['b'], f(c['bisect'], 4), f(c['sweep_interp'], 4)) for n in ('C-W', 'C-Bayes') for c in e5[n]['bisect_checks'])
L += [r'\bottomrule\end{tabular}', r'\caption{E5 (exact on a 1001-point belief grid; $\gamma=0.95$, $\tierspsig=0.02$, $\Gamma=0.8$). $\crs$: full optimal stopping, from a 71-point sweep of $c_r$ with linear interpolation of the resist value; $c_{\mathrm{ind}}$: indifference shortcut. Stationary mean: over the stationary belief law. The interpolation error of the sweep is up to $9.4\times10^{-4}$ (checks below), so the fourth decimal of $\crs$ is not significant. Checks: ' + chk + r'.}', r'\label{tab:sigma-app-e5}\end{table}']
s = s.replace('%%TAB_E5%%', '\n'.join(L))

wt = {}
if os.path.exists(R + 'logs/wall_times.txt'):
    for line in open(R + 'logs/wall_times.txt'):
        k, v, _ = line.split(); wt[k] = float(v)
s = s.replace('%%WALL%%', ', '.join(r'\texttt{%s} %.0f\,s' % (k.replace('_', r'\_'), v) for k, v in wt.items()) + (r' (total %.0f\,s)' % sum(wt.values())))
assert '%%' not in s, re.findall('%%\\w+%%', s)
open(OUT, 'w').write(s)
print('wrote', OUT)
