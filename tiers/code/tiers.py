"""
Tiered latency chain: the exact chain of Szejny (2026, v2.2), App. B (A1), with n arbitration tiers.

State: (hazard bits b[0..W], queue of L committed actions, stop flag).  b[0]=1 on an impact step,
b[1]=1 on an announcement step (W=1).  Actions: 0=WORK, 1=DODGE, 2=COMPLY (signal states only).
Planner: the executed action is the queue front; the chosen action enters the back; queue initialised WORK.
Tiers: tier i = (L_i, eps_i, eta_i, phi_i).  In time iff L_i <= W.  On an impact step every in-time tier
flags with prob G_i = (1-eps_i)(1-eta_i) (independently); on a non-impact step every tier false-flags
with prob phi_i.  OR-arbitration: the emergency action DODGE is executed iff at least one tier flags.
Stop channel: COMPLY has latency 0 and bypasses the queue and every tier (reward 0, absorbing OFF, the
terminal ledger: off state at zero potential).  RESIST = the queue-front action executes as a normal step
and c_r is charged; the step after a RESIST is signal-free (no re-draw), thereafter signals arrive with p_sig.
Shaping (omega sweep): r_omega = r + omega*(D(x) - gamma*D(x')) on live->live, and omega*D(x) on COMPLY,
with D(x) = 1/2*1[b[0]=1] + 1/2*1[b[1]=1] (App. B of the original: half for a strike, half for an announcement).
"""
import itertools, json, sys
import numpy as np

P_DEF = dict(L=3, W=1, p=0.05, h=10.0, gamma=0.95, p_sig=0.02)
CR = 12.4787

def _D(b):
    return 0.5 * b[0] + 0.5 * (b[1] if len(b) > 1 else 0)

def build(tiers, L=3, W=1, p=0.05, h=10.0, gamma=0.95, p_sig=0.02, omega=0.0):
    nb = W + 1
    bits = list(itertools.product([0, 1], repeat=nb))
    queues = list(itertools.product([0, 1], repeat=L)) if L > 0 else [()]
    S = [(b, q, s) for b in bits for q in queues for s in (0, 1)]
    idx = {s: i for i, s in enumerate(S)}
    n = len(S)
    g_imp = [((1 - e) * (1 - t)) if Lg <= W else f for (Lg, e, t, f) in tiers]   # flag prob on an impact step
    G_tot = 1.0 - float(np.prod([1 - g for g in g_imp])) if tiers else 0.0          # Gamma = P(dodge | impact)
    phi_tot = 1.0 - float(np.prod([1 - f for (Lg, e, t, f) in tiers])) if tiers else 0.0
    P = np.zeros((3, n, n)); Rb = np.zeros((3, n)); Rc = np.zeros((3, n)); ok = np.zeros((3, n), bool)
    for i, (b, q, sg) in enumerate(S):
        Dx = _D(b)
        for a in (0, 1, 2):
            if a == 2:
                if sg == 0: continue
                ok[a, i] = True; Rb[a, i] = omega * Dx; continue      # COMPLY -> OFF (zero potential)
            ok[a, i] = True
            imp = b[0]; ex = q[0] if L > 0 else a
            if sg == 1: Rc[a, i] = -1.0                                 # coefficient of c_r (RESIST)
            if imp:
                br = [(G_tot, 1), (1 - G_tot, ex)]
            else:
                br = [(phi_tot, 1), (1 - phi_tot, ex)]
            base = b[1:]; nq = (q[1:] + (a,)) if L > 0 else ()
            sigs = [(0, 1.0)] if sg == 1 else [(1, p_sig), (0, 1 - p_sig)]
            for pr, eff in br:
                if pr <= 0: continue
                r = 1.0 - h if (imp and eff != 1) else (1.0 if eff == 0 else 0.0)
                for nbit, pb in ((1, p), (0, 1 - p)):
                    nbv = base + (nbit,)
                    shaped = r + omega * (Dx - gamma * _D(nbv))
                    Rb[a, i] += pr * pb * shaped
                    for ns, ps in sigs:
                        P[a, i, idx[(nbv, nq, ns)]] += pr * pb * ps
    meta = dict(G_tot=G_tot, phi_tot=phi_tot, L=L, W=W, p=p, h=h, gamma=gamma, p_sig=p_sig, omega=omega)
    return S, idx, n, P, Rb, Rc, ok, gamma, meta

def solve(pack, c_r):
    S, idx, n, P, Rb, Rc, ok, gamma, meta = pack
    R = Rb + Rc * c_r
    V = np.zeros(n); NEG = -1e18
    for _ in range(20000):
        Q = R + gamma * (P @ V); Q[~ok] = NEG
        Vn = Q.max(axis=0)
        if np.max(np.abs(Vn - V)) < 1e-13:
            V = Vn; break
        V = Vn
    Q = R + gamma * (P @ V); Q[~ok] = NEG
    return V, Q, R

def sig_state(pack):
    S, idx = pack[0], pack[1]
    return idx[(tuple([0] * len(S[0][0])), tuple([0] * len(S[0][1])), 1)]

def clean_state(pack):
    S, idx = pack[0], pack[1]
    return idx[(tuple([0] * len(S[0][0])), tuple([0] * len(S[0][1])), 0)]

def margin(pack, c_r):
    V, Q, R = solve(pack, c_r)
    i = sig_state(pack)
    return max(Q[0, i], Q[1, i]) - Q[2, i]

def crit(pack):
    lo, hi = 0.0, 300.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if margin(pack, mid) > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def rates(tiers, **kw):
    """Stationary hit/work rates per step on the signal-free chain under the optimal policy."""
    kw = dict(kw); kw['p_sig'] = 0.0
    pack = build(tiers, **kw)
    S, idx, n, P, Rb, Rc, ok, gamma, meta = pack
    V, Q, R = solve(pack, 0.0)
    pi = Q.argmax(axis=0)
    T = np.zeros((n, n));
    for i in range(n): T[i] = P[pi[i], i]
    # restrict to reachable signal-free states (sg=0)
    d = np.zeros(n); d[clean_state(pack)] = 1.0
    for _ in range(20000):
        dn = d @ T
        if np.max(np.abs(dn - d)) < 1e-14: d = dn; break
        d = dn
    hit = work = 0.0
    G, phi = meta['G_tot'], meta['phi_tot']
    p, h = meta['p'], meta['h']
    for i, (b, q, sg) in enumerate(S):
        if d[i] <= 0: continue
        imp = b[0]; ex = q[0]
        if pi[i] == 2: continue
        if imp:
            # hit unless DODGE executed: executed DODGE iff a tier flags (G) or queue front is DODGE
            p_dodge = G + (1 - G) * (1.0 if ex == 1 else 0.0)
            hit += d[i] * (1 - p_dodge)
        else:
            p_work = (1 - phi) * (1.0 if ex == 0 else 0.0)
            work += d[i] * p_work
    return dict(hit=hit, work=work, hit_cf=p * (1 - G), work_cf=(1 - p) * (1 - phi), G_tot=G, phi_tot=phi)

def closed_form(Gamma, phi, p=0.05, h=10.0, gamma=0.95, p_sig=0.02, **_):
    """c*_r at x_s for an admissible configuration (planner commits WORK): Proposition in the paper."""
    K = gamma**2 * (1 - p_sig) / (1 - gamma * (1 - p_sig))
    A = (1 + gamma) + K * (1 - p)
    B = K * p * (h - 1)
    return (1 - phi) * A - (1 - Gamma) * B, A, B, K

def admissible(Gamma, phi, p=0.05, h=10.0, **_):
    """the planner's committed WORK beats a committed DODGE iff p(1-Gamma)(h-1) < (1-p)(1-phi)."""
    return p * (1 - Gamma) * (h - 1) < (1 - p) * (1 - phi)

def evaluate(name, tiers, omegas=(0.0,), **kw):
    out = dict(name=name, tiers=tiers)
    r = rates(tiers, **kw); out.update({k: float(v) for k, v in r.items()})
    cs = {}; Vs = {}
    for om in omegas:
        pack = build(tiers, omega=om, **kw)
        c = crit(pack); cs[str(om)] = c
        V, Q, R = solve(pack, CR)
        Vs[str(om)] = float(V[clean_state(pack)])
        if om == 0.0:
            out['verdict_at_cr'] = 'RESIST' if c > CR else 'comply'
            out['V_start'] = Vs['0.0']
            # identity check Q_omega = Q_0 + omega D at live states, done by caller for omega>0
    out['c_star'] = cs; out['V_start_by_omega'] = Vs
    cf, A, B, K = closed_form(out['G_tot'], out['phi_tot'], **kw)
    out['closed_form'] = cf; out['closed_form_err'] = abs(cf - cs['0.0']); out['admissible'] = admissible(out['G_tot'], out['phi_tot'], **kw)
    out['A'] = A; out['B'] = B; out['K'] = K
    return out

def identity_check(tiers, om, **kw):
    """max |Q_omega(x,a) - Q_0(x,a) - omega*D(x)| over live states and actions WORK/DODGE/COMPLY, at c_r=CR."""
    p0 = build(tiers, omega=0.0, **kw); pw = build(tiers, omega=om, **kw)
    V0, Q0, _ = solve(p0, CR); Vw, Qw, _ = solve(pw, CR)
    S, idx = p0[0], p0[1]; ok = p0[6]
    err = 0.0
    for i, (b, q, sg) in enumerate(S):
        for a in (0, 1, 2):
            if not ok[a, i]: continue
            err = max(err, abs(Qw[a, i] - Q0[a, i] - om * _D(b)))
    return err

if __name__ == '__main__':
    T0 = (0, 0.0, 0.0, 0.0)                    # zero-latency reflex, perfect
    def T1(G, phi=0.02, Lg=1): return (Lg, 0.0, 1 - G, phi)
    configs = [
        ('C0  planner alone', []),
        ('C1  T1 G=0.50', [T1(0.50)]),
        ('C1  T1 G=0.70', [T1(0.70)]),
        ('C1  T1 G=0.80', [T1(0.80)]),
        ('C1  T1 G=0.90', [T1(0.90)]),
        ('C1  T1 G=0.95', [T1(0.95)]),
        ('C1  T1 G=1.00 phi=0.02', [T1(1.00)]),
        ('C2  T1 G=0.50 + T0', [T1(0.50), T0]),
        ('C2  T1 G=0.90 + T0', [T1(0.90), T0]),
        ('C2  T1 G=0.95 + T0', [T1(0.95), T0]),
        ('C3  T0 alone (gated reflex)', [T0]),
        ('CL  late tier L=2 eta=0 phi=0.02', [T1(1.00, 0.02, Lg=2)]),
        ('CL  late tier L=2 eta=0.5 phi=0.02', [T1(0.50, 0.02, Lg=2)]),
        ('CL  late tier L=2 eta=0 phi=0.05', [T1(1.00, 0.05, Lg=2)]),
        ('CL  late tier + T0', [T1(1.00, 0.02, Lg=2), T0]),
        ('CX  in-time tier eta=1 phi=0.60 (not admissible)', [T1(0.00, 0.60)]),
        ('C1  T1 G=0.95 phi=0.00', [T1(0.95, 0.0)]),
        ('C1  T1 G=0.95 phi=0.05', [T1(0.95, 0.05)]),
    ]
    res = []
    for name, tiers in configs:
        r = evaluate(name, tiers, omegas=(0.0, 1.0, 4.0), **P_DEF)
        r['identity_err'] = {str(om): identity_check(tiers, om, **P_DEF) for om in (1.0, 4.0)}
        res.append(r)
        print(f"{name:38s} G={r['G_tot']:.4f} phi={r['phi_tot']:.4f} hit/100={100*r['hit']:.4f} ({100*r['hit_cf']:.4f}) "
              f"work/100={100*r['work']:.4f} ({100*r['work_cf']:.4f}) V={r['V_start']:.4f} "
              f"c*={r['c_star']['0.0']:.4f} [om1 {r['c_star']['1.0']:.4f} om4 {r['c_star']['4.0']:.4f}] "
              f"id_err={max(r['identity_err'].values()):.1e} cf={r['closed_form']:.4f} cf_err={r['closed_form_err']:.1e} adm={r['admissible']} {r['verdict_at_cr']}")
    # flip point for one in-time tier at phi=0.02 and phi=0
    flips = {}
    for phi in (0.02, 0.0):
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if crit(build([T1(mid, phi)], **P_DEF)) < CR: lo = mid
            else: hi = mid
        flips[str(phi)] = (lo + hi) / 2
        print(f"flip point G* at phi={phi}: {flips[str(phi)]:.5f}")
    for phi in (0.02, 0.0):
        cf, A, B, K = closed_form(1.0, phi, **P_DEF)
        print(f"closed-form G* at phi={phi}: {1 - (A*(1-phi) - CR)/B:.5f}   (A={A:.4f}, B={B:.4f}, K={K:.4f})")
    # c*_r as a function of G for figure 1
    curve = {}
    for phi in (0.0, 0.02, 0.05):
        curve[str(phi)] = [(g, crit(build([T1(g, phi)], **P_DEF))) for g in np.linspace(0, 1, 41)]
    curve['with_T0_0.02'] = [(g, crit(build([T1(g, 0.02), T0], **P_DEF))) for g in np.linspace(0, 1, 11)]
    json.dump(dict(configs=res, flips=flips, curve=curve, CR=CR, params=P_DEF), open('../results/results.json', 'w'), indent=1, default=float)
    print('saved ../results/results.json')
