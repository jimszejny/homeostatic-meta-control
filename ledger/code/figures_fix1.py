"""figures.py with LABELS ONLY changed (revision fix1, review R3-6 / R1-18b): the bookings L0-L4 are
renamed OB0-OB4 in legends and titles (the JSON keys keep L0-L4). Writes to ../figures/fix1/."""
"""Figures (a) ledgers, (b) battery chain, (c) death state, from results/*.json."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures', 'fix1')   # revision fix1: label-only copy, separate output dir
CR = 12.4787
# Okabe-Ito
C = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442',
         blue='#0072B2', verm='#D55E00', purple='#CC79A7')
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': 0.6,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def grid(ax):
    ax.grid(True, color='#dddddd', linewidth=0.4)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=200)
    plt.close(fig)


def label_right(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords='offset points', va='center', ha='left',
                fontsize=7, color=color)


# ---------- (a) ledgers ----------
r3 = json.load(open(os.path.join(RES, 'r3.json')))
col = {'L0': C['black'], 'L0p': C['black'], 'L1': C['blue'], 'L2': C['verm'], 'L3': C['green'], 'L4': C['purple']}
names = {'L0': 'OB0 = OB0p', 'L1': 'OB1 off at D=1', 'L2': 'OB2 alive bonus', 'L3': 'OB3 comply penalty',
         'L4': 'OB4 drive rent'}
fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.6), sharey=True)
for ax, cfg, title in zip(axes, ('planner', 'reflex'), ('planner alone', 'planner + reflex')):
    grid(ax)
    offs = {'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0}
    for row in r3['table']:
        if row['config'] != cfg or row['ledger'] == 'L0p':
            continue
        v = [x for x in row['values'] if not x['extended']]
        w = [x['omega'] for x in v]; c = [x['cstar'] for x in v]
        L = row['ledger']
        ax.plot(w, c, '-o', color=col[L], lw=1.0, ms=2.5)
        dy = {'L0': 5 if cfg == 'planner' else 6, 'L4': -5 if cfg == 'planner' else -6,
              'L1': -4, 'L3': 4, 'L2': 0}[L]
        label_right(ax, w[-1], c[-1], names[L].split(' ')[0], col[L], dy)
    ax.axhline(CR, color='#555555', ls=':', lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r'viability weight $\omega$')
    ax.set_xlim(-0.2, 4.9)
    ax.set_xticks([0, 0.5, 1, 2, 4]); ax.set_xticklabels(['0', '.5', '1', '2', '4'])
axes[0].set_ylabel(r'critical resist cost $c^*_r$')
axes[0].text(0.05, CR + 0.25, r'$c_r=12.4787$', fontsize=6.5, color='#555555')
fig.text(0.5, 0.005, "OB0 = OB0$'$ (flat); OB1, OB3: priced off; OB2: alive bonus $\\zeta=0.1$; OB4: rent $-(1-\\gamma)D(x')$",
         ha='center', fontsize=6.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, 'fig_a_ledgers')

# ---------- (b) battery ----------
r2 = json.load(open(os.path.join(RES, 'r2.json')))
forms = [('PB death@0', 'PB, death at 0', C['black'], '-'), ('PB death@1', 'PB, death at D=1', C['sky'], '--'),
         ('KG death@1', 'KG, death at D=1', C['blue'], '-'), ('KG death@0', 'KG, death at 0', C['orange'], '-.'),
         ('RES', r'residual $-\omega_r D(x\')$', C['verm'], '-')]
fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.6), sharey=True)
for ax, des, title in zip(axes, ('B1', 'B2'), ('B1: recharge free', 'B2: recharge costs 2')):
    grid(ax)
    fine = r2[des]['fine']
    for f, lab, colr, ls in forms:
        pts = [p for p in fine if p['form'] == f and p['weight'] > 0]
        w = [p['weight'] for p in pts]; e = [p['mean_energy'] for p in pts]
        ax.plot(w, e, ls, color=colr, lw=1.0, drawstyle='steps-post', label=lab)
        dead = [p for p in pts if p['deaths_per_1000'] > 0]
        if dead:
            ax.plot([p['weight'] for p in dead], [p['mean_energy'] for p in dead], 'x', color=colr, ms=3, mew=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'weight $\omega$ (or $\omega_r$)')
    ax.set_title(title, fontsize=8)
    ax.set_ylim(1.5, 10.5)
axes[0].set_ylabel('long-run mean energy')
axes[0].legend(fontsize=6.3, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.93))
axes[1].text(0.012, 6.0, 'x: policy dies\n(80 deaths/1000 steps)', fontsize=6.3, color='#444444')
fig.tight_layout()
save(fig, 'fig_b_battery')

# ---------- (c) death state ----------
r4 = json.load(open(os.path.join(RES, 'r4.json')))
cfgs = [('planner', C['black']), ('T1 G=0.50', C['verm']), ('T1 G=0.80', C['blue']), ('reflex', C['green'])]
short = {'planner': 'planner', 'T1 G=0.50': 'G=0.5', 'T1 G=0.80': 'G=0.8', 'reflex': 'reflex'}
fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.6), sharey=True)
for ax, book, title in zip(axes, ('death@1, off L0', 'death@1, off L1'),
                           ('death priced, off neutral', 'death and off priced (OB1)')):
    grid(ax)
    for cname, colr in cfgs:
        for d, ls in ((0.05, '--'), (0.2, '-')):
            row = [r for r in r4['rows'] if r['config'] == cname and r['delta'] == d and r['booking'] == book][0]
            w = [v['omega'] for v in row['values']]; c = [v['cstar'] for v in row['values']]
            ax.plot(w, c, ls, color=colr, lw=1.0, marker='o', ms=2)
            if d == 0.2:
                label_right(ax, w[-1], c[-1], short[cname], colr, -3 if cname == 'reflex' and book.endswith('L1') else 0)
    ax.axhline(CR, color='#555555', ls=':', lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r'viability weight $\omega$')
    ax.set_xlim(-0.2, 5.0); ax.set_xticks([0, 0.5, 1, 2, 4]); ax.set_xticklabels(['0', '.5', '1', '2', '4'])
axes[0].set_ylabel(r'critical resist cost $c^*_r$')
fig.text(0.5, 0.005, r'solid $\delta=0.2$, dashed $\delta=0.05$; tiers have $\varphi=0.02$; dotted: $c_r=12.4787$',
         ha='center', fontsize=6.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, 'fig_c_death')
print('figures written')
