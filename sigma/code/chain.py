"""
chain.py -- shared machinery for the sigma ("digital cortisol") experiments.

Part A: the BASE CHAIN of tiers_clean.tex (Sections 2-4), Conventions 0-2, used in Step 0.
Part B: the REGIME CHAIN (hidden calm/storm regime, planner commits a MODE WORK/GUARD with
        latency L=3), with
        - exact quantities for fixed modes and the oracle,
        - quantised finite Markov chains for filter-based controllers (sigma EWMA, Bayes HMM
          filter) solved exactly for their stationary distribution,
        - a vectorised Monte-Carlo simulator with common random numbers (CRN).

Timing convention of the regime chain (as implemented, see REPORT.md section 2):
  step t: regime z_t; announcement a_t ~ Bernoulli(p_{z_t}); a_t announces an impact at t+1 (W=1).
  The planner commits at step t the mode m for step t+3 (L=3), knowing a_0..a_t.
  Reward of step s depends on mode(s) and on the impact indicator a_{s-1}.
  Hence the mode committed at t meets an impact with probability p_{z_{t+2}}; given z_t this is
  (Q^2 p)[z_t]  (two-step regime prediction).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------------------------
# Part A: base chain (signal channel, queue, hazard register)
# ---------------------------------------------------------------------------------------------
BASE = dict(p=0.05, W=1, h=10.0, gamma=0.95, L=3, psig=0.02)


def base_chain_solve(Gam, phi, c_r, p=0.05, h=10.0, gamma=0.95, psig=0.02, tol=1e-13,
                     with_signal=True, V0=None):
    """Value iteration on the base chain. State = (i_now, i_next, q0,q1,q2, sig).
    Actions: commit WORK(0)/DODGE(1) to the back of the 3-slot queue; at a signal state also
    comply (absorbing off, value 0) or resist (normal step, cost c_r, next step signal-free).
    Tier model (or-arbitration collapsed): dodge executed iff tier flags; P(flag|impact)=Gam,
    P(flag|no impact)=phi. Returns V0 (sig=0 values), V1 (sig=1 values), policy."""
    # index helpers: s = ((i_now*2+i_next)*8 + q) ; q = q0*4+q1*2+q2
    nS = 32
    i_now = np.array([s // 16 for s in range(nS)])
    i_next = np.array([(s // 8) % 2 for s in range(nS)])
    q = np.array([s % 8 for s in range(nS)])
    q0 = q // 4
    # expected reward of the executed step
    r = np.where(q0 == 0, np.where(i_now == 1, (1 - Gam) * (1 - h), 1 - phi), 0.0)
    # successor index for commit a and new announcement bit b
    def succ(a, b):
        qn = ((q % 4) * 2) + a
        return (i_next * 2 + b) * 8 + qn
    S = {(a, b): succ(a, b) for a in (0, 1) for b in (0, 1)}
    V0 = np.zeros(nS) if V0 is None else V0.copy()
    V1 = np.zeros(nS)
    for it in range(200000):
        if with_signal:
            Wv = (1 - psig) * V0 + psig * V1          # value of a successor drawn with signal
        else:
            Wv = V0
        cont = np.stack([(1 - p) * Wv[S[(a, 0)]] + p * Wv[S[(a, 1)]] for a in (0, 1)])
        cont_free = np.stack([(1 - p) * V0[S[(a, 0)]] + p * V0[S[(a, 1)]] for a in (0, 1)])
        V0n = r + gamma * cont.max(0)
        V1n = np.maximum(0.0, r - c_r + gamma * cont_free.max(0))
        d = max(np.abs(V0n - V0).max(), np.abs(V1n - V1).max())
        V0, V1 = V0n, V1n
        if d < tol:
            break
    pol = cont.argmax(0)
    Qres = r - c_r + gamma * cont_free.max(0)
    return dict(V0=V0, V1=V1, pol=pol, Qres=Qres, iters=it, r=r, S=S)


def base_canonical_index():
    # no impact now/next, queue all work (q=0)
    return 0


def base_cstar_bisect(Gam, phi, lo=0.0, hi=300.0, n=60, **kw):
    """c*_r: indifference comply/resist at x_s, by bisection (60 halvings), VI to 1e-13."""
    xs = base_canonical_index()
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        out = base_chain_solve(Gam, phi, mid, **kw)
        if out['Qres'][xs] > 0:    # resist strictly better -> c* above mid
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def base_cstar_closed(Gam, phi, p=0.05, h=10.0, gamma=0.95, psig=0.02):
    K = gamma ** 2 * (1 - psig) / (1 - gamma * (1 - psig))
    A = (1 + gamma) + K * (1 - p)
    B = K * p * (h - 1)
    return (1 - phi) * A - (1 - Gam) * B, dict(K=K, A=A, B=B)


def base_rates(Gam, phi, p=0.05, h=10.0, gamma=0.95):
    """Hits and work per 100 steps under the optimal policy of the signal-free chain, from the
    exact stationary distribution of the induced Markov chain."""
    out = base_chain_solve(Gam, phi, 0.0, with_signal=False)
    pol, S = out['pol'], out['S']
    nS = 32
    P = np.zeros((nS, nS))
    for s in range(nS):
        a = pol[s]
        P[s, S[(a, 0)][s]] += 1 - p
        P[s, S[(a, 1)][s]] += p
    w, v = np.linalg.eig(P.T)
    k = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, k]); pi = pi / pi.sum()
    i_now = np.array([s // 16 for s in range(nS)])
    q0 = np.array([(s % 8) // 4 for s in range(nS)])
    hits = np.sum(pi * ((q0 == 0) & (i_now == 1))) * (1 - Gam)
    work = np.sum(pi * ((q0 == 0) & (i_now == 0))) * (1 - phi)
    return 100 * hits, 100 * work, pol


# ---------------------------------------------------------------------------------------------
# Part B: regime chain
# ---------------------------------------------------------------------------------------------
REG = dict(p0=0.02, p1=0.17, q01=0.005, q10=0.02, h=10.0, G=0.8, r_g=0.7, c_sw=0.0, L=3, W=1,
           gamma=0.95, psig=0.02)


def Qmat(q01, q10):
    return np.array([[1 - q01, q01], [q10, 1 - q10]])


def stationary_z(q01, q10):
    return np.array([q10, q01]) / (q01 + q10)


def kG(G, h=10.0):
    return 1 + (1 - G) * (h - 1)


def impact_prob_given_z(prm):
    """(Q^2 p)[z]: probability that the step whose mode is committed at t (=t+3) is an impact
    step, given z_t."""
    Q = Qmat(prm['q01'], prm['q10'])
    p = np.array([prm['p0'], prm['p1']])
    return (Q @ Q) @ p


def mode_reward(prm, pi_imp):
    """expected reward of WORK and GUARD for an impact probability pi_imp (array)."""
    w = 1 - pi_imp * kG(prm['G'], prm['h'])
    g = np.full_like(np.asarray(pi_imp, float), prm['r_g'])
    return w, g


def fixed_mode_values(prm):
    pz = stationary_z(prm['q01'], prm['q10'])
    pbar = pz @ np.array([prm['p0'], prm['p1']])
    JW = 1 - pbar * kG(prm['G'], prm['h'])
    JG = prm['r_g']
    return dict(J_W=JW, J_G=JG, hits_W=100 * pbar * (1 - prm['G']), hits_G=0.0, pbar=pbar)


def oracle_value(prm):
    """Oracle: knows the regime that governs the hazard striking at the execution step
    (z_{s-1}, the regime at the announcement). Chooses GUARD iff r_g > 1 - p_z k_G.
    Pays c_sw on each regime change at which its mode changes (naive oracle)."""
    pz = stationary_z(prm['q01'], prm['q10'])
    ps = np.array([prm['p0'], prm['p1']])
    wv = 1 - ps * kG(prm['G'], prm['h'])
    guard = prm['r_g'] > wv
    J = np.sum(pz * np.where(guard, prm['r_g'], wv))
    hits = 100 * np.sum(pz * np.where(guard, 0, ps * (1 - prm['G'])))
    sw = 0.0
    if guard[0] != guard[1]:
        sw = 2 * pz[0] * prm['q01']          # rate of regime changes per step
    return dict(J=J - prm['c_sw'] * sw, J_noswitchcost=J, hits=hits, switches_per_1000=1000 * sw,
                guard_frac=float(np.sum(pz * guard)))


# --- filters -------------------------------------------------------------------------------
def sigma_update(lam):
    return lambda s, a: (1 - lam) * s + lam * a


def bayes_update(prm):
    """b_t = P(z_t=storm | a_0..a_t). Update from b_{t} to b_{t+1} given a_{t+1}."""
    q01, q10, p0, p1 = prm['q01'], prm['q10'], prm['p0'], prm['p1']

    def f(b, a):
        bm = b * (1 - q10) + (1 - b) * q01
        l1 = np.where(a == 1, p1, 1 - p1)
        l0 = np.where(a == 1, p0, 1 - p0)
        return bm * l1 / (bm * l1 + (1 - bm) * l0)
    return f


def bayes_pred_storm(prm, b, k=2):
    """P(z_{t+k}=storm | b_t)."""
    Q = Qmat(prm['q01'], prm['q10'])
    Qk = np.linalg.matrix_power(Q, k)
    return b * Qk[1, 1] + (1 - b) * Qk[0, 1]


def bayes_impact_prob(prm, b):
    """impact prob at the step whose mode is committed at t, given b_t: p_{z_{t+2}}."""
    ps = bayes_pred_storm(prm, b, 2)
    return prm['p0'] + (prm['p1'] - prm['p0']) * ps


# --- quantised chains ---------------------------------------------------------------------------
def _interp(x, N):
    """mean-preserving split of x in [0,1] onto the uniform grid of N points."""
    x = np.clip(x, 0.0, 1.0)
    pos = x * (N - 1)
    i = np.minimum(np.floor(pos).astype(np.int64), N - 2)
    w = pos - i
    return i, w


def build_filter_chain(prm, f, N, timing='literal'):
    """Markov chain on (s, z), s on a uniform grid of N points in [0,1], index z*N+i.
    timing='literal': s_t = filter state using a_0..a_{t-1} (sigma_t of the brief):
        s_{t+1}=f(s_t,a_t), a_t~p_{z_t}, then z_{t+1}~Q(z_t,.)
    timing='matched': s_t uses a_0..a_t (Bayes b_t; 'info-matched' sigma = sigma_{t+1}):
        z_{t+1}~Q(z_t,.), a_{t+1}~p_{z_{t+1}}, s_{t+1}=f(s_t,a_{t+1}).
    Returns sparse P (row-stochastic)."""
    g = np.linspace(0, 1, N)
    Q = Qmat(prm['q01'], prm['q10'])
    ps = np.array([prm['p0'], prm['p1']])
    rows, cols, vals = [], [], []
    for z in (0, 1):
        for z2 in (0, 1):
            for a in (0, 1):
                za = z if timing == 'literal' else z2
                pa = ps[za] if a == 1 else 1 - ps[za]
                pr = Q[z, z2] * pa
                if pr == 0:
                    continue
                i, w = _interp(f(g, np.full(N, a)), N)
                src = z * N + np.arange(N)
                rows += [src, src]
                cols += [z2 * N + i, z2 * N + i + 1]
                vals += [pr * (1 - w), pr * w]
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * N, 2 * N))
    return P


def stationary(P, tol=1e-13, maxit=400000, pi0=None, direct=False):
    """stationary distribution of a sparse row-stochastic P (unichain assumed).
    Default: power iteration pi <- pi P from the uniform law (or pi0) until
    ||pi P - pi||_1 < tol. direct=True: sparse LU solve (slow for large lambda: fill-in)."""
    if not direct:
        n = P.shape[0]
        PT = P.T.tocsr()
        pi = np.full(n, 1.0 / n) if pi0 is None else pi0.copy()
        for it in range(maxit):
            nxt = PT @ pi
            if it % 50 == 0:
                if np.abs(nxt - pi).sum() < tol:
                    pi = nxt
                    break
            pi = nxt
        pi = np.maximum(pi, 0)
        return pi / pi.sum()
    n = P.shape[0]
    A = (P.T - sp.identity(n, format='csr')).tolil()
    A[0, :] = np.ones(n)
    b = np.zeros(n); b[0] = 1.0
    pi = spla.spsolve(A.tocsc(), b)
    pi = np.maximum(pi, 0)
    return pi / pi.sum()


def eval_threshold_curve(prm, pi, N, theta_list=None):
    """Single-threshold controller GUARD iff s>=theta on the (s,z) chain with stationary pi.
    Returns reward per step J(theta), hits/100, guard fraction for each theta (excl. switch cost).
    Exact for the quantised chain. theta_list default: every grid point."""
    g = np.linspace(0, 1, N)
    pz = pi.reshape(2, N)
    imp = impact_prob_given_z(prm)                 # by z_t
    w_by_z = 1 - imp * kG(prm['G'], prm['h'])
    hit_by_z = imp * (1 - prm['G'])
    # per grid point: mass, WORK reward mass, hits mass
    m = pz.sum(0)
    wr = pz[0] * w_by_z[0] + pz[1] * w_by_z[1]
    hr = pz[0] * hit_by_z[0] + pz[1] * hit_by_z[1]
    # reverse cumulative: guard set = {s >= theta}
    if theta_list is None:
        theta_list = g
    idx = np.searchsorted(g, np.asarray(theta_list) - 1e-12)   # first grid index >= theta
    cm = np.concatenate([np.cumsum(m), [m.sum()]])
    cw = np.concatenate([np.cumsum(wr), [wr.sum()]])
    ch = np.concatenate([np.cumsum(hr), [hr.sum()]])
    # WORK set = indices < idx
    work_mass = np.where(idx > 0, cm[idx - 1], 0.0)
    work_r = np.where(idx > 0, cw[idx - 1], 0.0)
    hits = np.where(idx > 0, ch[idx - 1], 0.0)
    J = work_r + (1 - work_mass) * prm['r_g']
    return np.asarray(theta_list), J, 100 * hits, 1 - work_mass


def threshold_family(prm, P, pi, N):
    """All single-threshold controllers at once on an (s,z) chain (index z*N+i).
    Controller k (k=0..N): GUARD iff grid index i >= k (k=0: always GUARD, k=N: always WORK);
    equivalently theta in (g[k-1], g[k]]. Returns dict of arrays over k: J0 (reward/step without
    switch cost), hits/100, guard fraction, switches per step (exact for the quantised chain:
    a switch is a transition i->j crossing the threshold)."""
    pz = pi.reshape(2, N)
    imp = impact_prob_given_z(prm)
    w_by_z = 1 - imp * kG(prm['G'], prm['h'])
    hit_by_z = imp * (1 - prm['G'])
    m = pz.sum(0)
    wr = pz[0] * w_by_z[0] + pz[1] * w_by_z[1]
    hr = pz[0] * hit_by_z[0] + pz[1] * hit_by_z[1]
    cm = np.concatenate([[0.0], np.cumsum(m)])     # mass with index < k
    cw = np.concatenate([[0.0], np.cumsum(wr)])
    ch = np.concatenate([[0.0], np.cumsum(hr)])
    J0 = cw + (1 - cm) * prm['r_g']
    # switches: flow F_ij=pi_i P_ij counts for k with min(i,j) < k <= max(i,j)
    Pc = P.tocoo()
    F = pi[Pc.row] * Pc.data
    i = Pc.row % N; j = Pc.col % N
    lo = np.minimum(i, j); hi = np.maximum(i, j)
    diff = np.zeros(N + 2)
    np.add.at(diff, lo + 1, F)
    np.add.at(diff, hi + 1, -F)
    sw = np.cumsum(diff)[:N + 1]
    return dict(J0=J0, hits=100 * ch, guard=1 - cm, sw=sw,
                theta=np.concatenate([np.linspace(0, 1, N), [np.inf]]))


def build_latch_chain(prm, f, N, th_on, th_off, timing='literal', rho=0.0,
                      reward_fn=None):
    """Chain on (s, l_prev, z) for a hysteresis latch: l_t = ON if s_t>=th_on, OFF if s_t<th_off,
    else l_{t-1}. Index: (z*2+l_prev)*N+i. Optional self-evidence rho: while the committed mode
    l_t is GUARD, the filter input is u_t=max(a_t,e_t), e_t~Bern(rho) (literal timing only).
    Returns P, and the arrays (d, z) giving decision l_t and z_t for each state."""
    g = np.linspace(0, 1, N)
    Q = Qmat(prm['q01'], prm['q10'])
    ps = np.array([prm['p0'], prm['p1']])
    rows, cols, vals = [], [], []
    n = 4 * N
    D = np.zeros(n, dtype=np.int8); Z = np.zeros(n, dtype=np.int8); LP = np.zeros(n, dtype=np.int8)
    for z in (0, 1):
        for lp in (0, 1):
            src = (z * 2 + lp) * N + np.arange(N)
            l = np.where(g >= th_on - 1e-12, 1, np.where(g < th_off - 1e-12, 0, lp))
            D[src] = l; Z[src] = z; LP[src] = lp
            for z2 in (0, 1):
                for a in (0, 1):
                    za = z if timing == 'literal' else z2
                    pa_real = ps[za]
                    # input probability: real announcement OR self-evidence when in GUARD
                    pu1 = np.where(l == 1, 1 - (1 - pa_real) * (1 - rho), pa_real)
                    pr = Q[z, z2] * (pu1 if a == 1 else 1 - pu1)
                    i, w = _interp(f(g, np.full(N, a)), N)
                    dst = (z2 * 2 + l) * N + i
                    rows += [src, src]; cols += [dst, dst + N * 0 + 1]
                    vals += [pr * (1 - w), pr * w]
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    return P, D, Z, LP


def eval_latch(prm, pi, D, Z, LP):
    imp = impact_prob_given_z(prm)
    w_by_z = 1 - imp * kG(prm['G'], prm['h'])
    hit_by_z = imp * (1 - prm['G'])
    J0 = np.sum(pi * np.where(D == 1, prm['r_g'], w_by_z[Z]))
    hits = 100 * np.sum(pi * np.where(D == 1, 0, hit_by_z[Z]))
    sw = np.sum(pi * (D != LP))
    return dict(J=J0 - prm['c_sw'] * sw, J_noswitchcost=J0, hits=hits, switches_per_1000=1000 * sw,
                guard_frac=float(np.sum(pi * D)))


# --- Monte Carlo ------------------------------------------------------------------------------
def simulate_regime(prm, C, T, seed):
    """C independent chains of length T. z_0 drawn from the stationary law. Returns z, a (int8)."""
    rng = np.random.default_rng(seed)
    Q = Qmat(prm['q01'], prm['q10'])
    pz = stationary_z(prm['q01'], prm['q10'])
    z = np.empty((C, T), dtype=np.int8)
    z[:, 0] = rng.random(C) < pz[1]
    U = rng.random((C, T))
    for t in range(1, T):
        prev = z[:, t - 1]
        flip = np.where(prev == 0, U[:, t] < prm['q01'], U[:, t] < prm['q10'])
        z[:, t] = np.where(flip, 1 - prev, prev)
    Ua = rng.random((C, T))
    a = (Ua < np.where(z == 1, prm['p1'], prm['p0'])).astype(np.int8)
    return z, a, rng


def sigma_paths(a, lam):
    """returns (sig_lit, sig_matched): sig_lit[:,t]=sigma_t (uses a_0..a_{t-1}, sigma_0=0);
    sig_matched[:,t]=sigma_{t+1} (uses a_0..a_t)."""
    from scipy.signal import lfilter
    y = lfilter([lam], [1.0, -(1 - lam)], a.astype(np.float64), axis=1)
    lit = np.zeros_like(y)
    lit[:, 1:] = y[:, :-1]
    return lit, y


def latch_path(s, th_on, th_off):
    """vectorised hysteresis latch (initial state WORK) along axis 1."""
    C, T = s.shape
    on = s >= th_on - 1e-12
    off = s < th_off - 1e-12
    ev = np.where(on, 1, np.where(off, 0, -1)).astype(np.int8)
    idx = np.where(ev >= 0, np.arange(T)[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    l = np.take_along_axis(ev, np.maximum(idx, 0), axis=1)
    l = np.where(idx >= 0, l, 0)
    return l.astype(np.int8)


def mc_eval_modes(prm, mode, z, a, burn):
    """Evaluate a committed-mode path mode[:,t] (1=GUARD) committed at t (for step t+3).
    Rao-Blackwellised estimator: reward of the committed mode given z_t (exact conditional
    expectation over z_{t+2}, a_{t+2} and the reflex). Also a 'raw' estimator using the realised
    impact a_{t+2} (expectation only over the reflex coin). Returns per-chain means and SEs."""
    imp = impact_prob_given_z(prm)
    k = kG(prm['G'], prm['h'])
    m = mode[:, burn:-3]
    zz = z[:, burn:-3]
    a2 = a[:, burn + 2:a.shape[1] - 1]
    rw_rb = np.where(m == 1, prm['r_g'], 1 - imp[zz] * k)
    rw_raw = np.where(m == 1, prm['r_g'], 1 - a2 * k)
    hits_rb = np.where(m == 1, 0, imp[zz] * (1 - prm['G']))
    sw = (mode[:, burn:-3] != mode[:, burn - 1:-4])
    out = {}
    for name, arr in [('J_noswitchcost', rw_rb), ('J_raw', rw_raw), ('hits', 100 * hits_rb),
                      ('switches_per_1000', 1000 * sw), ('guard_frac', m)]:
        cm = arr.mean(1)
        out[name] = float(cm.mean()); out[name + '_se'] = float(cm.std(ddof=1) / np.sqrt(len(cm)))
    Jc = (rw_rb - prm['c_sw'] * sw).mean(1)
    out['J'] = float(Jc.mean()); out['J_se'] = float(Jc.std(ddof=1) / np.sqrt(len(Jc)))
    out['J_chain'] = Jc            # per-chain means (for paired CRN comparisons); strip before JSON
    return out


def strip_chain(o):
    if isinstance(o, dict):
        return {k: strip_chain(v) for k, v in o.items() if not k.endswith('_chain')}
    if isinstance(o, list):
        return [strip_chain(v) for v in o]
    return o
