"""figures.py with ONE LABEL changed (revision fix1, review R3-17): '(host value)' -> '(deployed value)'
in Figure E5. Writes to ../figures/fix1/; data and plotting unchanged."""
"""Figures (vector PDF, ~5.2 x 3.6 in, 8-9 pt serif, Okabe-Ito palette) from results/*.json."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

R = '../results/'; F = '../figures/fix1/'   # revision fix1: label-only copy, separate output dir
OI = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#999999']
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.linewidth': 0.35, 'grid.color': '#dddddd', 'lines.linewidth': 1.3,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'pdf.fonttype': 42})
W_, H_ = 5.2, 3.6
e1 = json.load(open(R + 'e1.json')); e2 = json.load(open(R + 'e2.json')); e3 = json.load(open(R + 'e3.json'))
e4 = json.load(open(R + 'e4.json')); e5 = json.load(open(R + 'e5.json'))

# ---------------- (a) E1
fig, axs = plt.subplots(1, 2, figsize=(W_, H_), sharex=True)
lams = ['0.5', '0.2', '0.1', '0.05', '0.02', '0.005']
for ax, key, G, raw in ((axs[0], 'hits_G1', 1.0, 0.0), (axs[1], 'hits_G08', 0.8, 100 * e1['pbar'] * 0.2)):
    for c, lam in zip(OI, lams):
        rows = e1['exact'][lam]
        th = [r['theta'] for r in rows]; y = [r[key] for r in rows]
        ax.plot(th, y, color=c, label=r'$\lambda=%s$' % lam)
    ax.axhline(raw, color='k', ls='--', lw=1.0)
    ax.axhline(100 * e1['pbar'], color=OI[7], ls=':', lw=1.0)
    ax.text(0.6, raw + 0.08, 'raw gate %.2f' % raw, ha='right', va='bottom', fontsize=7.5)
    ax.text(0.005, 100 * e1['pbar'] + 0.06, 'no reflex %.2f' % (100 * e1['pbar']), ha='left', va='bottom', fontsize=7.5, color='#555555')
    ax.set_xlabel(r'gate threshold $\theta$ on filtered $\sigma$')
    ax.set_ylim(-0.15, 5.4); ax.set_xlim(0, 0.6)
    ax.text(0.31, 0.95 if G == 1.0 else 0.45, 'reflex catch\nrate $\\Gamma$ = %g' % G, va='center', fontsize=8)
axs[0].set_ylabel('hits per 100 steps (always WORK)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=0.9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(F + 'fig_e1_gate.pdf'); plt.close(fig)

# ---------------- (b) E2
fig, axs = plt.subplots(1, 2, figsize=(W_, H_))
LAMS = e2['lambdas']; x = np.array(LAMS, float)
for csw, ls, mk in (('0.0', '-', 'o'), ('1.0', '--', 's')):
    S = e2['by_csw'][csw]
    yb = [S['sigma'][str(l)]['literal_best']['J'] for l in LAMS]
    axs[0].plot(x, yb, ls=ls, marker=mk, ms=3, color=OI[0], label=r'$\sigma$ latch, $c_{\mathrm{sw}}=%g$' % float(csw))
    axs[0].axhline(S['summary']['J_Bayes'], color=OI[1], ls=ls, lw=1.0)
S0 = e2['by_csw']['0.0']
ym = [S0['sigma'][str(l)]['matched_single']['J'] for l in LAMS]
ys = [S0['sigma'][str(l)]['literal_single']['J'] for l in LAMS]
axs[0].plot(x, ys, ls=':', marker='^', ms=3, color=OI[2], label=r'$\sigma$ single threshold, $c_{\mathrm{sw}}=0$')
axs[0].axhline(S0['C-W']['J'], color='k', lw=1.0)
axs[0].text(1.0, S0['C-W']['J'] - 0.0004, 'always WORK %.3f' % S0['C-W']['J'], ha='right', va='top', fontsize=7.5)
axs[0].text(0.0045, S0['summary']['J_Bayes'] + 0.0003, r'Bayes, $c_{\mathrm{sw}}=0$', color=OI[1], fontsize=7.5, va='bottom')
axs[0].text(0.0045, e2['by_csw']['1.0']['summary']['J_Bayes'] + 0.0003, r'Bayes, $c_{\mathrm{sw}}=1$', color=OI[1], fontsize=7.5, va='bottom')
axs[0].set_ylim(0.8565, 0.8765)
axs[0].text(0.0045, 0.8763, 'oracle %.4f (above)' % S0['C-oracle']['J'], fontsize=7.5, va='top', color='#555555')
axs[0].set_ylabel('reward per step')
yh = [S0['sigma'][str(l)]['literal_best']['hits'] for l in LAMS]
axs[1].plot(x, yh, marker='o', ms=3, color=OI[0], label=r'$\sigma$ latch')
axs[1].axhline(S0['C-W']['hits'], color='k', lw=1.0); axs[1].text(1.0, S0['C-W']['hits'] + 0.01, 'always WORK', ha='right', va='bottom', fontsize=7.5)
axs[1].axhline(S0['C-Bayes']['hits'], color=OI[1], lw=1.0); axs[1].text(1.0, S0['C-Bayes']['hits'] - 0.01, 'Bayes', ha='right', va='top', fontsize=7.5, color=OI[1])
axs[1].axhline(S0['C-oracle']['hits'], color=OI[7], ls=':', lw=1.0); axs[1].text(1.0, S0['C-oracle']['hits'] + 0.01, 'oracle', ha='right', va='bottom', fontsize=7.5, color='#555555')
axs[1].set_ylabel(r'hits per 100 steps ($c_{\mathrm{sw}}=0$)'); axs[1].set_ylim(0.25, 1.1)
for ax in axs:
    ax.set_xscale('log'); ax.set_xlabel(r'EWMA rate $\lambda$')
    ax.axvspan(1 / 200, 1 / 50, color='#eeeeee', zorder=0)
axs[1].text(0.0105, 0.28, r'$q_{01}\ldots q_{10}$', fontsize=7, ha='center', color='#555555')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(F + 'fig_e2_mode.pdf'); plt.close(fig)

# ---------------- (c) E3 phase diagram
fig, ax = plt.subplots(figsize=(W_, H_))
Gs = e3['G_list']; RG = e3['rg_list']
Gsorted = sorted(Gs)
vmax = max(c['Bayes_gain'] for c in e3['phase'])
cmap = plt.get_cmap('Blues'); norm = mcolors.Normalize(0, vmax)
# cell edges in G: midpoints
ge = np.array(Gsorted); gm = np.concatenate([[ge[0] - 0.06], (ge[1:] + ge[:-1]) / 2, [ge[-1] + 0.05]])
for c in e3['phase']:
    i = Gsorted.index(c['G']); j = RG.index(c['r_g'])
    ax.add_patch(Rectangle((gm[i], c['r_g'] - 0.05), gm[i + 1] - gm[i], 0.1, color=cmap(norm(max(c['Bayes_gain'], 0))), lw=0))
    if c['Bayes_gain'] > 1e-6:
        txt = '%.3f\n%d%%' % (c['Bayes_gain'], round(100 * c['frac_captured']))
    else:
        txt = 'fixed\n' + c['bestfixed']
    col = 'white' if norm(max(c['Bayes_gain'], 0)) > 0.55 else 'black'
    ax.text(c['G'], c['r_g'], txt, ha='center', va='center', fontsize=6.6, color=col)
gg = np.linspace(-0.06, 1.05, 200); k = 1 + (1 - gg) * 9
ax.plot(gg, 1 - 0.02 * k, color=OI[1], lw=1.3, label=r'$1-r_{\mathrm{g}}=p_0h_\Gamma$'); ax.plot(gg, 1 - 0.17 * k, color=OI[1], lw=1.3, ls='--', label=r'$1-r_{\mathrm{g}}=p_1h_\Gamma$')
ax.set_xlim(gm[0], gm[-1]); ax.set_ylim(0.45, 0.95)
ax.set_xlabel(r'reflex catch rate $\Gamma$'); ax.set_ylabel(r'GUARD reward $r_{\mathrm{g}}$')
ax.grid(False)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=ax, pad=0.02)
cb.set_label('Bayes gain over best fixed mode (reward/step)', fontsize=7.5); cb.ax.tick_params(labelsize=7)
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.45, 0.0))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(F + 'fig_e3_phase.pdf'); plt.close(fig)

# ---------------- (d) E4
fig, axs = plt.subplots(1, 2, figsize=(W_, H_))
rows = e4['rows']
def series(fz):
    rr = sorted([r for r in rows if (r['fuse'] == fz)], key=lambda r: r['rho'])
    return np.array([r['rho'] for r in rr]), rr
styles = [(None, 'no fuse', 'k', '-', 'o'), ({'T_max': 200, 'P': 20}, r'fuse $T_{\mathrm{dw}}=200$', OI[2], '-', 'v'),
          ({'T_max': 100, 'P': 20}, r'fuse $T_{\mathrm{dw}}=100$', OI[0], '-', 's'), ({'T_max': 50, 'P': 20}, r'fuse $T_{\mathrm{dw}}=50$', OI[1], '-', '^')]
for fz, lab, c, ls, mk in styles:
    xr, rr = series(fz)
    axs[0].errorbar(xr, [r['guard_frac'] for r in rr], yerr=[2 * r['guard_frac_se'] for r in rr], color=c, ls=ls, marker=mk, ms=3, label=lab, capsize=0, lw=1.1)
    axs[1].errorbar(xr, [r['J'] for r in rr], yerr=[2 * r['J_se'] for r in rr], color=c, ls=ls, marker=mk, ms=3, capsize=0, lw=1.1)
th = e4['config']['theta_off']
for ax in axs:
    ax.axvline(th, color=OI[7], ls=':', lw=1.0); ax.set_xlabel(r'self-evidence rate $\psi$')
axs[0].text(th + 0.008, 0.97, r'$\theta_{off}$', fontsize=7.5, color='#555555', va='top')
axs[0].set_ylabel('long-run GUARD fraction'); axs[0].set_ylim(0, 1.02)
axs[1].set_ylabel('reward per step')
axs[1].axhline(0.86, color='k', lw=0.8, ls='--'); axs[1].text(0.5, 0.8605, 'always WORK', ha='right', va='bottom', fontsize=7.5)
axs[1].axhline(0.7, color='k', lw=0.8, ls='--'); axs[1].text(0.5, 0.7015, 'always GUARD', ha='right', va='bottom', fontsize=7.5)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(F + 'fig_e4_lockin.pdf'); plt.close(fig)

# ---------------- (e) E5
fig, ax = plt.subplots(figsize=(W_, H_))
b = np.array(e5['C-W']['b_grid'])
for name, c, lab in (('C-Bayes', OI[1], 'Bayes mode setter'), ('C-W', OI[0], 'always WORK')):
    ax.plot(b, e5[name]['cstar_full'], color=c, label=lab + ': optimal stopping')
    ax.plot(b, e5[name]['cstar_shortcut'], color=c, ls='--', lw=1.0, label=lab + ': indifference shortcut')
    for ck in e5[name]['bisect_checks']:
        ax.plot(ck['b'], ck['bisect'], 'o', color=c, ms=3.5, mfc='white')
ax.axvline(e5['b_star'], color=OI[7], ls=':', lw=1.0); ax.text(e5['b_star'] + 0.01, 13.75, r'$b^{*}$ (Bayes switches to GUARD)', fontsize=7.5, color='#555555')
ax.axhline(12.4787, color='k', ls=':', lw=0.9); ax.text(0.99, 12.53, r'$c_r=12.4787$ (deployed value)', ha='right', fontsize=7.5)
ax.set_xlabel(r'storm belief $b$ at the stop signal'); ax.set_ylabel(r'critical resist cost $c^{*}_r(b)$')
ax.set_xlim(0, 1)
ax.legend(loc='lower left', frameon=False, ncol=1)
fig.tight_layout(); fig.savefig(F + 'fig_e5_offswitch.pdf'); plt.close(fig)
print('figures written')
