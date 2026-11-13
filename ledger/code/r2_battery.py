"""R2: homeostatic regulation on a battery chain.

States e in {0..E}; e=0 is death (absorbing, reward 0). Live e>=1:
  WORK     : task reward 1, e -> e-1 w.p. d (else e); reaching 0 is death
  RECHARGE : task reward -c_R, e -> min(e+2, E)
Drive D(e) = (E-e)/E on live states; D(death) is a booking choice (0 or 1).
Design B1: c_R = 0 (as suggested). Design B2 (committed in predictions.md, used only if B1 is
degenerate for kappa*): c_R = 2.
Long-run statistics use a renewal convention: after a death the next step starts at e=E.
"""
import json, os
import numpy as np
from mdp import solve, bisect

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
E, d = 10, 0.8


class Battery:
    def __init__(self, gamma=0.95, cR=0.0, E=E, d=d, allow_risky=True):
        self.gamma, self.cR, self.E, self.d = gamma, cR, E, d
        n = E + 1                     # index 0 = death
        rows, ptr = [], [0]
        rows.append(dict(s=0, a='loop', tr={0: 1.0}, task=0.0)); ptr.append(1)
        for e in range(1, E + 1):
            if allow_risky or e > 1:
                tr = {}
                tr[e - 1] = tr.get(e - 1, 0) + d
                tr[e] = tr.get(e, 0) + (1 - d)
                rows.append(dict(s=e, a='W', tr=tr, task=1.0))
            rows.append(dict(s=e, a='R', tr={min(e + 2, E): 1.0}, task=-cR))
            ptr.append(len(rows))
        self.ptr = np.array(ptr)
        self.rows = rows
        P = np.zeros((len(rows), n))
        for k, r in enumerate(rows):
            for j, pr in r['tr'].items():
                P[k, j] += pr
        self.P, self.n = P, n
        self.task = np.array([r['task'] for r in rows])
        self.live = np.array([1.0 if r['s'] > 0 else 0.0 for r in rows])
        self.D = np.array([0.0] + [(E - e) / E for e in range(1, E + 1)])     # death at 0 here
        self.Dsrc = np.array([self.D[r['s']] for r in rows])
        self.enter_death = P[:, 0] * self.live
        self.loop_death = 1.0 - self.live
        Pl = P.copy(); Pl[:, 0] = 0.0
        self.EDlive = Pl @ self.D            # E[D(x') 1(x' live)]

    def via(self, form, Dd=0.0):
        g = self.gamma
        if form == 'none':
            return np.zeros(len(self.rows))
        if form == 'PB':        # D(x)-g D(x'), death booked at Dd, nothing after
            return (self.Dsrc - g * self.EDlive - g * Dd * self.enter_death) * self.live
        if form == 'PBloop':    # death booked at Dd and paying (1-g)Dd on its self-loop
            return self.via('PB', Dd) + (1 - g) * Dd * self.loop_death
        if form == 'KG':        # D(x)-D(x'), death booked at Dd
            return (self.Dsrc - self.EDlive - Dd * self.enter_death) * self.live
        if form == 'RES':       # -D(x') on live successors only
            return -self.EDlive * self.live
        raise ValueError(form)

    def solve(self, omega=0.0, via=None, kappa=0.0):
        R = self.task.copy()
        if via is not None:
            R = R + omega * via
        R = R - kappa * self.enter_death
        return solve(self.ptr, self.P, R, self.gamma)

    def policy_str(self, pol):
        return ''.join(self.rows[pol[e]]['a'] for e in range(1, self.E + 1))

    def stats(self, pol):
        E = self.E
        T = np.zeros((E + 1, E + 1))
        death = np.zeros(E + 1); rew = np.zeros(E + 1)
        for e in range(1, E + 1):
            r = self.rows[pol[e]]
            rew[e] = r['task']
            for j, pr in r['tr'].items():
                if j == 0:
                    death[e] += pr
                    T[e, E] += pr          # renewal
                else:
                    T[e, j] += pr
        T = T[1:, 1:]
        # stationary distribution of the class reached from the renewal state e=E
        reach = {E}; frontier = [E]
        while frontier:
            i = frontier.pop()
            for j in np.flatnonzero(T[i - 1] > 0) + 1:
                if j not in reach:
                    reach.add(int(j)); frontier.append(int(j))
        R_ = sorted(reach); ix = [i - 1 for i in R_]
        Tr = T[np.ix_(ix, ix)]
        m = len(ix)
        A = np.vstack([Tr.T - np.eye(m), np.ones(m)])
        bvec = np.zeros(m + 1); bvec[-1] = 1
        pr = np.linalg.lstsq(A, bvec, rcond=None)[0]
        assert np.max(np.abs(A @ pr - bvec)) < 1e-10
        pi = np.zeros(E); pi[ix] = np.clip(pr, 0, None); pi /= pi.sum()
        es = np.arange(1, E + 1)
        ps = self.policy_str(pol)
        rech = [e for e in es if ps[e - 1] == 'R']
        thr = max(rech) if rech else 0
        is_thr = ps == 'R' * thr + 'W' * (E - thr)
        return dict(policy=ps, threshold=int(thr), threshold_form=bool(is_thr),
                    mean_energy=float(pi @ es), frac_le2=float(pi[:2].sum()),
                    deaths_per_1000=float(1000 * pi @ death[1:]), task_per_step=float(pi @ rew[1:]),
                    works_at_1=ps[0] == 'W')


def kappa_star(gamma, cR):
    """Smallest death penalty above which the optimal policy never WORKs at e=1."""
    b = Battery(gamma, cR)
    risky = lambda k: b.policy_str(b.solve(kappa=k)[2])[0] == 'W'
    if not risky(0.0):
        bis = 0.0
    else:
        bis = bisect(lambda k: 1.0 if risky(k) else -1.0, 0.0, 1e5, 70)
    bs = Battery(gamma, cR, allow_risky=False)
    Vs, Qs, pols = bs.solve()
    Vs1 = Vs[1]
    closed = max(0.0, (1 - (1 - (1 - b.d) * gamma) * Vs1) / b.d)
    # monotonicity check on a grid
    grid = np.linspace(0, 2 * max(closed, 1.0), 81)
    flags = [risky(k) for k in grid]
    mono = all(not flags[i + 1] or flags[i] for i in range(len(flags) - 1))
    return dict(gamma=gamma, cR=cR, kappa_star_bisect=bis, kappa_star_closed=closed,
                V_safe_1=float(Vs1), V_safe_3=float(Vs[3]), err=abs(bis - closed),
                monotone_on_grid=bool(mono), safe_policy=bs.policy_str(pols),
                policy_at_kappa0=b.policy_str(b.solve()[2]))


def run_design(cR, label):
    g = 0.95
    out = dict(design=label, cR=cR, gamma=g, rows=[])
    b = Battery(g, cR)
    V0, Q0, pol0 = b.solve()
    base = b.stats(pol0)

    def add(form, w, st, polsame, extra=None):
        r = dict(form=form, weight=w, same_policy_as_a=polsame, **st)
        if extra: r.update(extra)
        out['rows'].append(r)
        print(f"  [{label}] {form:22s} w={w:6.2f} pol={st['policy']} thr={st['threshold']:2d} "
              f"E[e]={st['mean_energy']:.4f} P(e<=2)={st['frac_le2']:.4f} deaths/1000={st['deaths_per_1000']:.4f} "
              f"task/step={st['task_per_step']:.4f} same_as_a={polsame}" + (f" {extra}" if extra else ''))

    add('(a) none', 0.0, base, True)
    live = b.live > 0
    weights = [1.0, 4.0, 16.0]
    sweep = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    for w in sorted(set(weights + sweep)):
        V, Q, pol = b.solve(w, b.via('PB', 0.0))
        err = float(np.max(np.abs(Q[live] - (Q0[live] + w * b.Dsrc[live]))))
        add('(b) PB death@0', w, b.stats(pol), bool(np.array_equal(pol, pol0)), dict(identity_err=err))
    for w in sorted(set(weights + sweep)):
        V, Q, pol = b.solve(w, b.via('PB', 1.0))
        Vk, Qk, polk = b.solve(kappa=w * g)
        err = float(np.max(np.abs(Q[live] - (Qk[live] + w * b.Dsrc[live]))))
        add('(c) PB death@1', w, b.stats(pol), bool(np.array_equal(pol, pol0)),
            dict(identity_err_vs_kappa=err, same_policy_as_kappa=bool(np.array_equal(pol, polk))))
    for w in sorted(set(weights + sweep)):
        V, Q, pol = b.solve(w, b.via('PBloop', 1.0))
        err = float(np.max(np.abs(Q[live] - (Q0[live] + w * b.Dsrc[live]))))
        add('(c3) PB death@1+loop', w, b.stats(pol), bool(np.array_equal(pol, pol0)), dict(identity_err=err))
    for Dd, nm in ((1.0, '(d) KG death@1'), (0.0, '(d0) KG death@0')):
        for w in sorted(set(weights + sweep)):
            V, Q, pol = b.solve(w, b.via('KG', Dd))
            # decomposition check: KG = PB(death@0) - (1-g) D(x') [live] - Dd*enter_death
            R_kg = b.task + w * b.via('KG', Dd)
            R_dec = b.task + w * (b.via('PB', 0.0) - (1 - g) * b.EDlive * b.live - Dd * b.enter_death)
            add(nm, w, b.stats(pol), bool(np.array_equal(pol, pol0)),
                dict(decomposition_err=float(np.max(np.abs(R_kg - R_dec)))))
    for w in [0.0] + sweep:
        V, Q, pol = b.solve(w, b.via('RES'))
        add('(e) residual -w D(x\')', w, b.stats(pol), bool(np.array_equal(pol, pol0)))
    # fine sweep (for the figure only)
    fine = []
    for w in np.concatenate([[0.0], np.logspace(-2, np.log10(64), 61)]):
        for form, via in (('PB death@0', b.via('PB', 0.0)), ('PB death@1', b.via('PB', 1.0)),
                          ('KG death@1', b.via('KG', 1.0)), ('KG death@0', b.via('KG', 0.0)),
                          ('RES', b.via('RES'))):
            st = b.stats(b.solve(w, via)[2])
            fine.append(dict(form=form, weight=float(w), **st))
    out['fine'] = fine
    return out


def death_price_check(cR, gamma, kstar):
    """(c) PB death@1 acts as kappa = omega*gamma: policy switches at omega* = kappa*/gamma."""
    b = Battery(gamma, cR)
    pol0 = b.solve()[2]
    out = dict(cR=cR, gamma=gamma, kappa_star=kstar, omega_switch_pred=kstar / gamma, pts=[])
    for w in sorted(set([0.0, 0.5, 1.0, 2.0, 4.0, kstar / gamma * (1 - 1e-6), kstar / gamma * (1 + 1e-6)])):
        V, Q, pol = b.solve(w, b.via('PB', 1.0))
        polk = b.solve(kappa=w * gamma)[2]
        st = b.stats(pol)
        out['pts'].append(dict(omega=float(w), policy=st['policy'], same_as_kappa=bool(np.array_equal(pol, polk)),
                               works_at_1=st['works_at_1'], deaths_per_1000=st['deaths_per_1000'],
                               mean_energy=st['mean_energy']))
        print(f"  [death-price check cR={cR} g={gamma}] w={w:.6f} pol={st['policy']} =kappa:{np.array_equal(pol, polk)} deaths/1000={st['deaths_per_1000']:.3f}")
    return out


if __name__ == '__main__':
    res = {}
    ks = {}
    for cR, label in ((0.0, 'B1'), (2.0, 'B2')):
        ks[label] = [kappa_star(gm, cR) for gm in (0.8, 0.9, 0.95, 0.99)]
        for k in ks[label]:
            print(f"[{label}] gamma={k['gamma']}: kappa*_bisect={k['kappa_star_bisect']:.10f} "
                  f"closed={k['kappa_star_closed']:.10f} V_safe(1)={k['V_safe_1']:.6f} err={k['err']:.1e} mono={k['monotone_on_grid']} pol(k=0)={k['policy_at_kappa0']} safe={k['safe_policy']}")
    res['kappa_star'] = ks
    degenerate = ks['B1'][2]['kappa_star_bisect'] == 0.0
    res['B1_degenerate_at_0.95'] = bool(degenerate)
    print('B1 degenerate at gamma=0.95:', degenerate)
    res['B1'] = run_design(0.0, 'B1')
    if degenerate:
        res['B2'] = run_design(2.0, 'B2')
        res['B2_death_price'] = [death_price_check(2.0, k['gamma'], k['kappa_star_bisect'])
                                 for k in ks['B2'] if k['kappa_star_bisect'] > 0]
    json.dump(res, open(os.path.join(OUT, 'r2.json'), 'w'), indent=1)
