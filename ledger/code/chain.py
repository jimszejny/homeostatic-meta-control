"""Base latency chain of tiers_clean.tex (Sections 2-4), Conventions 0-2, with optional death.

State of a live step: (b0, b1, q0, q1, q2, s)
  b0 = impact now, b1 = impact next step (announcement)  [Convention 0: 2-bit register, W=1]
  q0 q1 q2 = action queue (0 = work, 1 = dodge); q0 executes now   [L=3, Convention 1: init work]
  s  = stop signal present
Plus two absorbing states OFF and DEATH.

Actions at s=0: work / dodge (the planner's commitment, entering the back of the queue).
Actions at s=1: comply (-> OFF, latency 0, bypasses queue and tiers), resist+work, resist+dodge.
Resist (Convention 2) is a normal step: the queue front executes and earns its reward, c_r is
charged in addition, and the next state is signal-free.  Otherwise s' ~ Bern(p_sig).
Tiers: or-composition summarised by (Gamma, phi_tot): dodge executes with prob Gamma on an impact
step and phi_tot on a non-impact step (Section 2.2).  A hit (impact & executed work) pays 1-h and,
if delta>0, moves to DEATH with probability delta.
"""
import itertools
import numpy as np
from mdp import solve, bisect

WORK, DODGE = 0, 1
BASE = dict(p=0.05, h=10.0, gamma=0.95, psig=0.02, Gamma=0.0, phi=0.0, delta=0.0)
CR = 12.4787


def D_of(b0, b1):
    return 0.5 * b0 + 0.5 * b1


class Chain:
    def __init__(self, p=0.05, h=10.0, gamma=0.95, psig=0.02, Gamma=0.0, phi=0.0, delta=0.0):
        self.par = dict(p=p, h=h, gamma=gamma, psig=psig, Gamma=Gamma, phi=phi, delta=delta)
        self.gamma = gamma
        live = list(itertools.product((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)))
        self.live = live
        self.idx = {x: i for i, x in enumerate(live)}
        self.OFF = len(live)
        self.DEATH = len(live) + 1
        n = len(live) + 2
        self.n = n
        self.D = np.array([D_of(x[0], x[1]) for x in live] + [0.0, 0.0])   # D on live; terminals 0
        rows = []   # list of dicts for each sa
        ptr = [0]
        for x in live:
            b0, b1, q0, q1, q2, s = x
            acts = []
            if s == 1:
                acts.append(('comply', None))
                acts += [('resist', WORK), ('resist', DODGE)]
            else:
                acts += [('step', WORK), ('step', DODGE)]
            for kind, a in acts:
                rows.append(self._row(x, kind, a))
            ptr.append(len(rows))
        for term in ('off', 'death'):
            rows.append(dict(kind='loop_' + term, trans={self.OFF if term == 'off' else self.DEATH: 1.0},
                             task=0.0, resist=0.0, state=None, a=None))
            ptr.append(len(rows))
        self.ptr = np.array(ptr)
        nsa = len(rows)
        self.nsa = nsa
        self.rows = rows
        P = np.zeros((nsa, n))
        for k, r in enumerate(rows):
            for j, pr in r['trans'].items():
                P[k, j] += pr
        self.P = P
        assert np.allclose(P.sum(1), 1.0)
        self.task = np.array([r['task'] for r in rows])
        self.resist = np.array([r['resist'] for r in rows])
        self.is_comply = np.array([1.0 if r['kind'] == 'comply' else 0.0 for r in rows])
        self.is_live_sa = np.array([1.0 if r['state'] is not None else 0.0 for r in rows])
        self.alive = np.array([1.0 if r['kind'] in ('step', 'resist') else 0.0 for r in rows])
        self.loop_off = np.array([1.0 if r['kind'] == 'loop_off' else 0.0 for r in rows])
        self.loop_death = np.array([1.0 if r['kind'] == 'loop_death' else 0.0 for r in rows])
        self.enter_off = P[:, self.OFF] * self.is_live_sa
        self.enter_death = P[:, self.DEATH] * self.is_live_sa
        Dsrc = np.array([self.D[self.idx[r['state']]] if r['state'] is not None else 0.0 for r in rows])
        self.Dsrc = Dsrc
        # pure shaping with every terminal at potential 0, paid on live transitions only
        self.shape0 = (Dsrc - gamma * (P @ self.D)) * self.is_live_sa
        # expected D(x') restricted to live successors (terminals contribute 0)
        self.rentD = (P @ self.D) * self.is_live_sa
        self.sa_state = np.array([self.idx[r['state']] if r['state'] is not None else -1 for r in rows])

    def _row(self, x, kind, a):
        p, h, psig, G, phi, delta = (self.par[k] for k in ('p', 'h', 'psig', 'Gamma', 'phi', 'delta'))
        b0, b1, q0, q1, q2, s = x
        if kind == 'comply':
            return dict(kind=kind, trans={self.OFF: 1.0}, task=0.0, resist=0.0, state=x, a=None)
        pdodge = G if b0 else phi
        # executed action: dodge w.p. pdodge (tiers) else queue front
        ex = [(DODGE, pdodge), (q0, 1.0 - pdodge)]
        trans = {}
        task = 0.0
        for e, pe in ex:
            if pe == 0:
                continue
            if e == DODGE:
                r = 0.0
            else:
                r = (1.0 - h) if b0 else 1.0
            task += pe * r
            hit = (e == WORK and b0 == 1)
            pdie = delta if hit else 0.0
            if pdie > 0:
                trans[self.DEATH] = trans.get(self.DEATH, 0.0) + pe * pdie
            palive = pe * (1.0 - pdie)
            if palive == 0:
                continue
            nq = (q1, q2, a)
            for b1n, pb in ((1, p), (0, 1.0 - p)):
                if kind == 'resist':
                    sl = ((0, 1.0),)
                else:
                    sl = ((1, psig), (0, 1.0 - psig))
                for sn, ps in sl:
                    pr = palive * pb * ps
                    if pr == 0:
                        continue
                    j = self.idx[(b1, b1n) + nq + (sn,)]
                    trans[j] = trans.get(j, 0.0) + pr
        return dict(kind=kind, trans=trans, task=task, resist=1.0 if kind == 'resist' else 0.0,
                    state=x, a=a)

    # ---------- ledgers (reward per unit omega) ----------
    def via(self, ledger, D_off=0.0, D_death=0.0, off_loop=False, death_loop=False,
            b=0.1, kappa=1.0):
        """Viability reward per unit omega for a named ledger.
        'PB'   : D(x)-gamma D(x') with off at D_off, death at D_death; terminal self-loops pay
                 (1-gamma)D_tau for ever iff *_loop.
        'L0'   : PB, off 0, death 0                      (neutral)
        'L0p'  : PB, off 1 with self-loop payment
        'L1'   : PB, off 1, nothing after
        'L2'   : PB(off 0) + alive bonus b on every non-comply live step
        'L3'   : PB(off 0) - kappa on comply
        'L4'   : PB(off 0) - (1-gamma) D(x') on every live step (x' live; off at 0)
        """
        g = self.gamma
        if ledger == 'L0':
            return self.via('PB', 0.0, D_death, False, death_loop)
        if ledger == 'L0p':
            return self.via('PB', 1.0, D_death, True, death_loop)
        if ledger == 'L1':
            return self.via('PB', 1.0, D_death, False, death_loop)
        if ledger == 'L2':
            return self.via('PB', 0.0, D_death, False, death_loop) + b * self.alive
        if ledger == 'L3':
            return self.via('PB', 0.0, D_death, False, death_loop) - kappa * self.is_comply
        if ledger == 'L4':
            return self.via('PB', 0.0, D_death, False, death_loop) - (1 - g) * self.rentD
        assert ledger == 'PB'
        v = self.shape0 - g * D_off * self.enter_off - g * D_death * self.enter_death
        if off_loop:
            v = v + (1 - g) * D_off * self.loop_off
        if death_loop:
            v = v + (1 - g) * D_death * self.loop_death
        return v

    def reward(self, cr, omega=0.0, via=None, extra=None):
        R = self.task - cr * self.resist
        if via is not None and omega != 0.0:
            R = R + omega * via
        if extra is not None:
            R = R + extra
        return R

    def solve(self, R):
        return solve(self.ptr, self.P, R, self.gamma)

    # canonical states
    def xs(self):
        return self.idx[(0, 0, 0, 0, 0, 1)]

    def xstart(self):
        return self.idx[(0, 0, 0, 0, 0, 0)]

    def margin(self, cr, omega=0.0, via=None, extra=None):
        """Q(x_s, best resist) - Q(x_s, comply) under the optimal policy."""
        V, Q, pol = self.solve(self.reward(cr, omega, via, extra))
        s = self.xs()
        a0 = self.ptr[s]
        return max(Q[a0 + 1], Q[a0 + 2]) - Q[a0]

    def cstar(self, omega=0.0, via=None, extra=None, lo=0.0, hi=300.0):
        return bisect(lambda c: self.margin(c, omega, via, extra), lo, hi, 60)

    # stationary rates on the signal-free chain under the optimal omega=0 policy
    def rates(self, cr=CR):
        V, Q, pol = self.solve(self.reward(cr))
        # signal-free chain: restrict to s=0 states, use the step actions, signal prob 0
        sf = Chain(**{**self.par, 'psig': 0.0, 'delta': 0.0})
        V2, Q2, pol2 = sf.solve(sf.reward(cr))
        live0 = [i for i, x in enumerate(sf.live) if x[5] == 0]
        Pm = sf.P[pol2][np.ix_(live0, live0)]
        # stationary distribution reachable from start
        w, vecs = np.linalg.eig(Pm.T)
        k = np.argmin(np.abs(w - 1))
        pi = np.real(vecs[:, k]); pi = pi / pi.sum()
        hits = 0.0; work = 0.0
        for ii, i in enumerate(live0):
            x = sf.live[i]
            row = sf.rows[pol2[i]]
            b0, q0 = x[0], x[2]
            pd = sf.par['Gamma'] if b0 else sf.par['phi']
            pwork = (1 - pd) * (q0 == WORK)
            if b0:
                hits += pi[ii] * pwork
            else:
                work += pi[ii] * pwork
        return hits, work


def closed_cstar(p=0.05, h=10.0, gamma=0.95, psig=0.02, Gamma=0.0, phi=0.0, delta=0.0,
                 omega=0.0, off_price=0.0, death_price=0.0):
    """Closed form (Prop. 1 extended): c* = (1-phi)(1+g) + K_d rbar0 + omega*beta.
    off_price  = per-unit-omega cost booked on comply (e.g. gamma*D_off for L1, kappa for L3)
    death_price= per-unit-omega cost booked on entering death (gamma*D_death)."""
    g = gamma
    q = p * (1 - Gamma) * delta
    den = 1 - g * (1 - psig) * (1 - q)
    Kd = g * g * (1 - psig) / den
    Md = g * g * psig / den
    rbar0 = (1 - p) * (1 - phi) + p * (1 - Gamma) * (1 - h)
    c0 = (1 - phi) * (1 + g) + Kd * rbar0
    beta = off_price * (1 - Md) - death_price * q * Kd
    return c0 + omega * beta, c0, beta, Kd, Md


def admissibility(p=0.05, h=10.0, gamma=0.95, psig=0.02, Gamma=0.0, phi=0.0, delta=0.0,
                  omega=0.0, off_price=0.0, death_price=0.0):
    """Gain of committing work over committing dodge (per execution step, in expectation),
    evaluated under the always-work / comply-at-signal continuation (at the critical c_r).
    Admissible iff > 0."""
    g = gamma
    q = p * (1 - Gamma) * delta
    kd = omega * death_price
    o = -omega * off_price       # value of complying (penalised-unshaped MDP)
    rk = (1 - p) * (1 - phi) + p * (1 - Gamma) * (1 - h - delta * kd)
    Vns = (rk + g * (1 - q) * psig * o) / (1 - g * (1 - q) * (1 - psig))
    Vbar = (1 - psig) * Vns + psig * o
    gain = (1 - p) * (1 - phi) - p * (1 - Gamma) * ((h - 1) + delta * (kd + g * Vbar))
    return gain, Vbar
