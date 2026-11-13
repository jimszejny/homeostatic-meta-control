"""E5: the off-switch price of stress. Regime chain + the base chain's stop channel (Convention 2:
resist is a normal step charged c_r, next step signal-free; comply -> absorbing off, reward 0).
The mode policy is fixed (Bayes myopic, c_sw=0, or always-WORK); the only decision is comply/resist
at signal states. State = (belief b on a grid, hazard register (i_now,i_next), 3-slot mode queue, signal).
Discounted, gamma=0.95. c*_r(b) at the canonical signal state x_s(b) = (b, no impact now/next,
queue filled with m(b), signal).
  - 'shortcut' c_ind(b): Proposition-1 logic, continuation = reward stream until the next signal
    (every later signal state complies).
  - 'full' c*(b): root in c_r of Q_resist(x_s(b); c_r)=0 with the optimal stopping rule at every
    later signal state (value iteration to 1e-11); obtained on all b from a c_r sweep with linear
    interpolation of Q_resist, and verified by bisection (45 halvings) at three beliefs."""
import json, time
import numpy as np
from chain import *
from chain import _interp
from e2_mode import bayes_threshold_c0

t0 = time.time()
NB = 1001


class Model:
    def __init__(self, prm, mode_fn, NB=NB):
        self.prm = prm; self.NB = NB
        g = np.linspace(0, 1, NB); self.g = g
        f = bayes_update(prm)
        bm = g * (1 - prm['q10']) + (1 - g) * prm['q01']
        self.pa1 = prm['p0'] + (prm['p1'] - prm['p0']) * bm
        self.ib = {}; self.w = {}
        for a in (0, 1):
            self.ib[a], self.w[a] = _interp(f(g, np.full(NB, a)), NB)
        self.m = mode_fn(g).astype(int)                     # committed mode per belief
        s = np.arange(32); i_now = s // 16; i_next = (s // 8) % 2; q = s % 8; q0 = q // 4
        self.S = {a: ((i_next * 2 + a) * 8)[None, :] + ((q % 4) * 2)[None, :] + self.m[:, None] for a in (0, 1)}
        G, h, rg = prm['G'], prm['h'], prm['r_g']
        r1 = np.where(q0 == 0, np.where(i_now == 1, (1 - G) * (1 - h), 1.0), rg)
        self.r = np.broadcast_to(r1, (NB, 32)).copy()
        self.xs = 0 * 8 + (self.m * 7)      # canonical: i_now=i_next=0, queue all m(b): q = 0 or 7

    def P(self, X):
        """E[X(next)] for X of shape (NB,32)."""
        out = 0
        for a in (0, 1):
            pa = self.pa1 if a == 1 else 1 - self.pa1
            ib, w = self.ib[a], self.w[a]
            S = self.S[a]
            out = out + pa[:, None] * ((1 - w)[:, None] * X[ib[:, None], S] + w[:, None] * X[ib[:, None] + 1, S])
        return out

    def shortcut(self, gamma, psig, tol=1e-12):
        U = np.zeros((self.NB, 32))
        for _ in range(100000):
            Un = self.r + gamma * (1 - psig) * self.P(U)
            if np.abs(Un - U).max() < tol:
                U = Un; break
            U = Un
        cont = self.r + gamma * self.P(U)
        return cont[np.arange(self.NB), self.xs]

    def solve(self, c_r, gamma, psig, tol=1e-11, V0=None, V1=None):
        V0 = np.zeros((self.NB, 32)) if V0 is None else V0
        V1 = np.zeros((self.NB, 32)) if V1 is None else V1
        for it in range(100000):
            PV0 = self.P(V0)
            V0n = self.r + gamma * self.P((1 - psig) * V0 + psig * V1)
            V1n = np.maximum(0.0, self.r - c_r + gamma * PV0)
            d = max(np.abs(V0n - V0).max(), np.abs(V1n - V1).max())
            V0, V1 = V0n, V1n
            if d < tol:
                break
        Qres = self.r - c_r + gamma * self.P(V0)
        return V0, V1, Qres[np.arange(self.NB), self.xs]


def full_curve(M, cgrid, gamma, psig):
    Qs = []; V0 = V1 = None
    for c in cgrid:
        V0, V1, q = M.solve(c, gamma, psig, V0=V0, V1=V1)
        Qs.append(q)
    Qs = np.array(Qs)        # [c, b], decreasing in c
    cst = np.full(M.NB, np.nan)
    for j in range(M.NB):
        col = Qs[:, j]
        k = np.nonzero((col[:-1] > 0) & (col[1:] <= 0))[0]
        if len(k):
            k = k[0]; c0, c1 = cgrid[k], cgrid[k + 1]
            cst[j] = c0 + (c1 - c0) * col[k] / (col[k] - col[k + 1])
    return cst


def bisect(M, j, gamma, psig, lo, hi, n=45):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        _, _, q = M.solve(mid, gamma, psig)
        if q[j] > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


res = {}
base = dict(REG)
gamma, psig = base['gamma'], base['psig']
# --- validation: with p0=p1=0.05 the regime is irrelevant -> Proposition 1 closed form
val = {}
for G in (0.8, 0.0, 1.0):
    prm = dict(base, p0=0.05, p1=0.05, G=G)
    M = Model(prm, lambda g: np.zeros_like(g), NB=101)
    cind = M.shortcut(gamma, psig)
    cb = bisect(M, 50, gamma, psig, 0.0, 40.0)
    cf, _ = base_cstar_closed(G, 0.0)
    val[str(G)] = dict(shortcut=float(cind[50]), bisect_full=cb, closed_form_prop1=cf)
    print('validation G=%g' % G, val[str(G)], flush=True)
res['validation_p0_eq_p1'] = val

bstar = bayes_threshold_c0(base)
ctrls = {'C-W': lambda g: np.zeros_like(g), 'C-Bayes': lambda g: (g >= bstar).astype(float)}
# stationary belief law for averaging
Pb = build_filter_chain(base, bayes_update(base), NB, 'matched')
pib = stationary(Pb).reshape(2, NB).sum(0)
res['b_star'] = bstar
for name, fn in ctrls.items():
    M = Model(base, fn)
    cind = M.shortcut(gamma, psig)
    lo, hi = float(np.nanmin(cind)) - 0.3, float(np.nanmax(cind)) + 3.0
    cgrid = np.linspace(lo, hi, 71)
    cfull = full_curve(M, cgrid, gamma, psig)
    checks = []
    for bq in (0.05, 0.3, 0.9):
        j = int(round(bq * (NB - 1)))
        cb = bisect(M, j, gamma, psig, lo - 1, hi + 1)
        checks.append(dict(b=bq, bisect=cb, sweep_interp=float(cfull[j]), shortcut=float(cind[j])))
        print(name, checks[-1], flush=True)
    d = np.diff(cfull)
    res[name] = dict(b_grid=M.g.tolist(), cstar_full=cfull.tolist(), cstar_shortcut=cind.tolist(),
                     bisect_checks=checks, c_sweep=[lo, hi, len(cgrid)],
                     stationary_avg_full=float(np.nansum(pib * cfull)), stationary_avg_shortcut=float(np.sum(pib * cind)),
                     full_nonincreasing=bool(np.all(d <= 1e-9)), full_max_increase=float(d.max()),
                     shortcut_nonincreasing=bool(np.all(np.diff(cind) <= 1e-9)), shortcut_max_increase=float(np.diff(cind).max()),
                     full_minus_shortcut_max=float(np.nanmax(cfull - cind)), full_minus_shortcut_min=float(np.nanmin(cfull - cind)),
                     at=dict((str(b), dict(full=float(cfull[int(round(b * (NB - 1)))]), shortcut=float(cind[int(round(b * (NB - 1)))])))
                             for b in (0.0, 0.033, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.61, 0.7, 0.9, 1.0)))
    for key, y in (('full', cfull), ('shortcut', cind)):
        i = int(np.nonzero(y < 12.4787)[0][0])
        res[name]['b_cross_cr_12_4787_' + key] = [float(M.g[i - 1]), float(M.g[i])]   # resist below, comply above
    res[name]['b_argmin_full_minus_shortcut'] = float(M.g[np.nanargmin(cfull - cind)])
    print(name, 'avg full', res[name]['stationary_avg_full'], 'avg shortcut', res[name]['stationary_avg_shortcut'],
          'nonincr', res[name]['full_nonincreasing'], res[name]['full_max_increase'], f'{time.time()-t0:.0f}s', flush=True)
res['stationary_belief'] = dict(mean=float(np.sum(pib * np.linspace(0, 1, NB))))
res['wall_s'] = time.time() - t0
json.dump(res, open('../results/e5.json', 'w'), indent=1, default=float)
print(f'E5 done {time.time()-t0:.1f}s')
