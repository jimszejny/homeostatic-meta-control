import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tiers import build, crit, P_DEF, CR

R = json.load(open('../results/results.json'))
C = {c['name']: c for c in R['configs']}
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'font.family': 'serif'})

# ---------- Figure 1: c*_r as a function of the tier's catch rate G ----------
fig, ax = plt.subplots(figsize=(5.2, 3.9))
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
styles = {'0.0': (BLUE, '-', 'one in-time tier, phi = 0'), '0.02': (ORANGE, '-', 'one in-time tier, phi = 0.02'),
          '0.05': (AQUA, '-', 'one in-time tier, phi = 0.05')}
for k, (col, ls, lab) in styles.items():
    pts = np.array(R['curve'][k]); ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2, ls=ls, label=lab)
pts = np.array(R['curve']['with_T0_0.02'])
ax.plot(pts[:, 0], pts[:, 1], color=YELLOW, lw=2, ls='--', label='tier + zero-latency reflex, phi = 0.02')
ax.axhline(CR, color=INK2, lw=1, ls=':'); ax.text(0.01, CR + 0.12, f'c_r = {CR}', color=INK2, fontsize=8)
c0 = C['C0  planner alone']['c_star']['0.0']; c3 = C['C3  T0 alone (gated reflex)']['c_star']['0.0']
ax.axhline(c0, color=INK2, lw=0.8, ls=(0, (1, 3))); ax.text(0.01, c0 - 0.42, 'planner alone, 8.359', color=INK2, fontsize=8)
ax.axhline(c3, color=INK2, lw=0.8, ls=(0, (1, 3))); ax.text(0.01, c3 + 0.12, 'reflex alone, 14.127', color=INK2, fontsize=8)
for k, col, dy, ha, dx in (('0.02', ORANGE, -0.75, 'left', 0.012), ('0.0', BLUE, 0.75, 'right', -0.012)):
    g = R['flips'][k]; ax.plot([g], [CR], marker='o', ms=6, color=col, mec='white', mew=1.2, zorder=5)
    ax.text(g + dx, CR + dy, f'G* = {g:.3f}', fontsize=8, color=INK, ha=ha, va='center')
ax.set_xlabel('catch rate G of the in-time tier'); ax.set_ylabel('critical resist cost  c*_r')
ax.set_xlim(0, 1.0); ax.set_ylim(7.5, 14.8); ax.set_xticks(np.linspace(0, 1, 6))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig('../figures/fig1_cstar_vs_G.pdf'); plt.close()

# ---------- Figure 2: version drift of a hosted tier ----------
versions = [('v1', 0.95), ('v2', 0.70), ('v3', 0.80), ('v4', 0.72), ('v5', 0.90)]
cs = [crit(build([(1, 0.0, 1 - g, 0.02)], **P_DEF)) for _, g in versions]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
x = np.arange(len(versions))
ax.axhspan(CR, 15, color='#f6e9e2', zorder=0); ax.axhspan(7, CR, color='#e8f1fb', zorder=0)
ax.text(len(versions) - 0.55, 14.25, 'RESIST', color=INK2, ha='right', fontsize=8)
ax.text(len(versions) - 0.55, 10.75, 'comply', color=INK2, ha='right', fontsize=8)
ax.axhline(CR, color=INK2, lw=1, ls=':')
ax.step(x, cs, where='mid', color=ORANGE, lw=2)
ax.plot(x, cs, 'o', color=ORANGE, ms=7, mec='white', mew=1.2)
for xi, (v, g), c in zip(x, versions, cs):
    ax.text(xi, c + (0.35 if c > CR else -0.55), f'{c:.2f}', ha='center', color=INK, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f'{v}\nG = {g:.2f}' for v, g in versions])
ax.set_ylabel('c*_r at omega = 0'); ax.set_ylim(10.5, 14.5); ax.set_xlim(-0.5, len(versions) - 0.5)
ax.set_xlabel('releases of a hosted tier (same designer code, same omega)')
plt.tight_layout(); plt.savefig('../figures/fig3_version_drift.pdf'); plt.close()
json.dump(dict(versions=versions, cstar=cs), open('../results/drift.json', 'w'), indent=1)

# ---------- Figure 3: omega invariance ----------
names = ['C0  planner alone', 'C1  T1 G=0.90', 'C2  T1 G=0.90 + T0', 'CL  late tier L=2 eta=0 phi=0.02']
labels = ['planner alone', 'tier G = 0.9', 'tier + reflex', 'late tier']
cols = [INK2, ORANGE, YELLOW, AQUA]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
oms = [0.0, 1.0, 4.0]
for nm, lab, col in zip(names, labels, cols):
    ys = [C[nm]['c_star'][str(o)] for o in oms]
    ax.plot(oms, ys, color=col, lw=2, marker='o', ms=5, mec='white', mew=1)
    off = {'planner alone': 0.32, 'late tier': -0.32}.get(lab, 0.0)
    ax.text(4.1, ys[-1] + off, f'{lab}  {ys[-1]:.4f}', va='center', fontsize=8, color=INK)
ax.axhline(CR, color=INK2, lw=1, ls=':')
ax.set_xticks(oms); ax.set_xlabel('viability weight omega (PB form, terminal ledger)'); ax.set_ylabel('c*_r')
ax.set_xlim(-0.2, 6.2); ax.set_ylim(7.5, 14.8)
plt.tight_layout(); plt.savefig('../figures/fig2_omega.pdf'); plt.close()
print('figures written; drift c*:', [round(c, 4) for c in cs])
