"""POST HOC (added in revision after review R1, finding 1 and finding 9). Not pre-specified.

Scope of Theorem thm:death (closed form (death-cstar)). Its proof needs the canonical signal state
x_s to have the largest RESIST margin among all signal states at c_r = (death-cstar). R1 reports that
this fails for the booking D(OFF)=1, D(DEAD)=0 (off state priced, death at zero potential) at large
omega. This script, using only chain.py / mdp.py of this package:

 (1) planner alone, delta=0.2, D(OFF)=1, D(DEAD)=0: bisection c*_r versus (death-cstar) for omega up
     to 1000; at the closed-form root, the RESIST margin (optimal policy) at every signal state;
     the omega at which the largest margin over signal states other than x_s reaches zero
     (bisection in omega), i.e. where the closed form stops being exact;
 (2) the same threshold for T1 G=0.5, T1 G=0.8 (phi=0.02, delta=0.2) and the planner at delta=0.05;
 (3) the other three bookings (D(OFF) <= D(DEAD)) on the grid's five controllers and three deltas
     at omega in {0,4,16,64,256,1000} wherever (death-adm) holds: max |bisection - closed form| and
     the smallest gap between the margin at x_s and the largest margin elsewhere;
 (4) the slope identities of the corollary: beta_delta(off neutral, death priced) = -gamma nu K_delta
     and beta_delta(both priced) = gamma(1-gamma)(1+gamma psig+gamma nu(1-psig))/(1-gamma(1-psig)(1-nu)),
     checked against closed_cstar on a grid of nu, with the values at nu = 0, 0.05 (= p) and 1.
Writes ../results/posthoc_r1_scope.json.
"""
import json, os
import numpy as np
from chain import Chain, closed_cstar, admissibility

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'posthoc_r1_scope.json')
g = 0.95


def margins_at(ch, cr, w, via):
    """RESIST margin (best resist minus comply) at every signal state under the optimal policy."""
    V, Q, pol = ch.solve(ch.reward(cr, w, via))
    out = {}
    for i, x in enumerate(ch.live):
        if x[5] != 1:
            continue
        a0 = ch.ptr[i]
        out[x] = float(max(Q[a0 + 1], Q[a0 + 2]) - Q[a0])
    return out


def other_max(ch, cr, w, via):
    m = margins_at(ch, cr, w, via)
    xs = (0, 0, 0, 0, 0, 1)
    best = max((v, k) for k, v in m.items() if k != xs)
    return best[0], best[1], m[xs]


def run_case(G, phi, d, Doff, Ddead, w):
    ch = Chain(Gamma=G, phi=phi, delta=d)
    via = ch.via('PB', D_off=Doff, D_death=Ddead)
    cf = closed_cstar(Gamma=G, phi=phi, delta=d, omega=w, off_price=g * Doff, death_price=g * Ddead)[0]
    cs = ch.cstar(w, via, lo=-500.0, hi=5000.0)
    gain = admissibility(Gamma=G, phi=phi, delta=d, omega=w, off_price=g * Doff, death_price=g * Ddead)[0]
    mo, xo, mxs = other_max(ch, cf, w, via)
    return dict(omega=w, cstar_bisect=cs, closed=cf, err=cs - cf, adm_gain=gain,
                max_margin_other_at_closed_root=mo, argmax_other=list(xo), margin_xs_at_closed_root=mxs)


def threshold(G, phi, d, Doff, Ddead, lo=0.0, hi=3000.0, n=50):
    """smallest omega at which some signal state other than x_s has a non-negative margin at the
    closed-form root (bisection; assumes one crossing)."""
    ch = Chain(Gamma=G, phi=phi, delta=d)
    via = ch.via('PB', D_off=Doff, D_death=Ddead)

    def f(w):
        cf = closed_cstar(Gamma=G, phi=phi, delta=d, omega=w, off_price=g * Doff, death_price=g * Ddead)[0]
        return other_max(ch, cf, w, via)[0]
    if f(hi) < 0:
        return None
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


res = {'note': 'POST HOC, added in revision after review R1 (findings 1 and 9); not pre-specified'}

# (1) the failing booking, planner alone, delta = 0.2
res['planner_d0.2_offpriced_death0'] = [run_case(0.0, 0.0, 0.2, 1.0, 0.0, w)
                                        for w in (4.0, 100.0, 200.0, 240.0, 245.0, 300.0, 1000.0)]
# (2) thresholds
res['threshold_offpriced_death0'] = {
    'planner d=0.2': threshold(0.0, 0.0, 0.2, 1.0, 0.0),
    'T1 G=0.5 d=0.2': threshold(0.5, 0.02, 0.2, 1.0, 0.0),
    'T1 G=0.8 d=0.2': threshold(0.8, 0.02, 0.2, 1.0, 0.0),
    'planner d=0.05': threshold(0.0, 0.0, 0.05, 1.0, 0.0),
}
# (3) the three bookings with D(OFF) <= D(DEAD), within admissibility
CONFIGS = [('planner', 0.0, 0.0), ('T1 G=0.50', 0.5, 0.02), ('T1 G=0.80', 0.8, 0.02),
           ('T1 G=0.95', 0.95, 0.02), ('reflex', 1.0, 0.0)]
BOOK = [('death@0, off L0', 0.0, 0.0), ('death@1, off L0', 1.0, 0.0), ('death@1, off L1', 1.0, 1.0),
        ('death@0, off L1', 0.0, 1.0)]
grid = []
for cname, G, phi in CONFIGS:
    for d in (0.0, 0.05, 0.2):
        for bname, Dd, Do in BOOK:
            for w in (0.0, 4.0, 16.0, 64.0, 256.0, 1000.0):
                r = run_case(G, phi, d, Do, Dd, w)
                r.update(config=cname, delta=d, booking=bname, admissible=r['adm_gain'] > 0)
                grid.append(r)
summ = {}
for bname, Dd, Do in BOOK:
    rows = [r for r in grid if r['booking'] == bname and r['admissible']]
    summ[bname] = dict(n_admissible_points=len(rows),
                       max_abs_err=max(abs(r['err']) for r in rows),
                       min_gap_xs_minus_other=min(r['margin_xs_at_closed_root'] - r['max_margin_other_at_closed_root'] for r in rows),
                       n_points_other_state_nonneg=sum(r['max_margin_other_at_closed_root'] >= 0 for r in rows))
res['grid_extended'] = grid
res['grid_extended_summary'] = summ

# (4) slope identities
p, psig = 0.05, 0.02
chk = []
for nu in np.linspace(0, 1, 1001):
    den = 1 - g * (1 - psig) * (1 - nu)
    K, M = g * g * (1 - psig) / den, g * g * psig / den
    b_neutral = -g * nu * K
    b_both = g * (1 - M) - g * nu * K
    b_both_formula = g * (1 - g) * (1 + g * psig + g * nu * (1 - psig)) / den
    chk.append(abs(b_both - b_both_formula))
# closed_cstar uses nu = delta p (1-Gamma); reach nu in [0, p] through Gamma=0, delta in [0,1]
def beta_both(nu):
    return closed_cstar(Gamma=0.0, delta=nu / p, off_price=g, death_price=g)[2]
def beta_neutral(nu):
    return closed_cstar(Gamma=0.0, delta=nu / p, death_price=g)[2]
res['slope_identity'] = dict(max_abs_identity_error_nu_grid_1001=max(chk),
                             beta_both_nu0=beta_both(0.0), beta_both_nu_p=beta_both(p),
                             beta_both_nu1_formula=g * (1 - g) * (1 + g * psig + g * (1 - psig)) / (1 - 0.0),
                             beta_neutral_nu_p=beta_neutral(p),
                             min_beta_both_over_nu_in_0_p=min(beta_both(x) for x in np.linspace(0, p, 501)))

json.dump(res, open(OUT, 'w'), indent=1, default=float)
print('(1) planner, delta=0.2, D(OFF)=1, D(DEAD)=0')
for r in res['planner_d0.2_offpriced_death0']:
    print(f"  w={r['omega']:7.1f} bisect={r['cstar_bisect']:.5f} closed={r['closed']:.5f} err={r['err']:.2e} "
          f"adm={r['adm_gain']:.3f} max other margin={r['max_margin_other_at_closed_root']:+.4f} at {r['argmax_other']}")
print('(2) thresholds', res['threshold_offpriced_death0'])
print('(3)', json.dumps(summ, indent=1))
print('(4)', res['slope_identity'])
print('saved', OUT)
