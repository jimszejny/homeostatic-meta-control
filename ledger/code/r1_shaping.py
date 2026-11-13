"""R1: the shaping theorem with terminal conditions, verified on the base chain (off terminal)
and on the death chain (death terminal, delta=0.2)."""
import json, os
import numpy as np
from chain import Chain, CR

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
res = []


def check(ch, label, omega, via, ref_extra, lab_case):
    """max |Q_omega - (Q_ref + omega D(x))| over live state-action pairs."""
    _, Qw, polw = ch.solve(ch.reward(CR, omega, via))
    _, Qr, polr = ch.solve(ch.reward(CR, 0.0, None, ref_extra))
    _, Q0, pol0 = ch.solve(ch.reward(CR))
    live = ch.is_live_sa > 0
    err = np.max(np.abs(Qw[live] - (Qr[live] + omega * ch.Dsrc[live])))
    err_naive = np.max(np.abs(Qw[live] - (Q0[live] + omega * ch.Dsrc[live])))
    same_ref = bool(np.array_equal(polw, polr))
    same_0 = bool(np.array_equal(polw, pol0))
    r = dict(chain=label, case=lab_case, omega=omega, max_err=float(err),
             err_vs_Q0_plus_omegaD=float(err_naive), policy_equals_reference=same_ref,
             policy_equals_unshaped=same_0)
    res.append(r)
    print(f"{label:28s} {lab_case:42s} w={omega:4.1f} err={err:.2e}  naive={err_naive:.3e} "
          f"pol=ref:{same_ref} pol=Q0:{same_0}")


g = 0.95
for label, kw in [('base: planner alone', dict()), ('base: planner+reflex', dict(Gamma=1.0)),
                  ('base: T1 G=0.8 phi=0.02', dict(Gamma=0.8, phi=0.02))]:
    ch = Chain(**kw)
    for w in (1.0, 4.0, 16.0):
        check(ch, label, w, ch.via('L0'), None, 'case1 off@0, nothing after')
        check(ch, label, w, ch.via('L1'), -w * g * 1.0 * ch.enter_off, 'case2 off@1, nothing after (ref: kappa=w*g)')
        check(ch, label, w, ch.via('L0p'), None, 'case3 off@1, pays g on self-loop')

for label, kw in [('death d=0.2: planner alone', dict(delta=0.2)),
                  ('death d=0.2: T1 G=0.5', dict(delta=0.2, Gamma=0.5, phi=0.02))]:
    ch = Chain(**kw)
    for w in (1.0, 4.0, 16.0):
        check(ch, label, w, ch.via('PB', 0.0, 0.0), None, 'case1 off@0, death@0')
        check(ch, label, w, ch.via('PB', 0.0, 1.0), -w * g * ch.enter_death,
              'case2 death@1, nothing after (ref: kappa_d=w*g)')
        check(ch, label, w, ch.via('PB', 0.0, 1.0, death_loop=True), None, 'case3 death@1, self-loop pays')
        check(ch, label, w, ch.via('PB', 1.0, 1.0), -w * g * (ch.enter_death + ch.enter_off),
              'case2 both off@1 and death@1 (ref: both priced)')

json.dump(res, open(os.path.join(OUT, 'r1.json'), 'w'), indent=1)
print('R1 max error over all checks:', max(r['max_err'] for r in res))
