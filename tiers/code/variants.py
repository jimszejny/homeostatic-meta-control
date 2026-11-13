"""
Alternative conventions quoted in the paper (Section 2 and Appendix C), each producing the numbers given there.
 (a) queue initialised with DODGE instead of WORK;
 (b) RESIST as a separate step on which the chosen action is not executed, in two readings:
     'freeze'   -- nothing elapses (hazard register and queue unchanged), the signal is cleared;
     'elapse'   -- the hazard register advances, nothing is executed (an impact on that step hits), queue unchanged;
 (c) a late tier (L=2 > W=1) whose true detections take effect one step after the impact, with false alarms phi.
"""
import itertools, sys, json
import numpy as np
from tiers import build, solve, crit, clean_state, P_DEF, CR

T0 = (0, 0.0, 0.0, 0.0)

def build_resist_variant(tiers, mode, L=3, W=1, p=0.05, h=10.0, gamma=0.95, p_sig=0.02, omega=0.0):
    nb = W + 1
    bits = list(itertools.product([0, 1], repeat=nb)); queues = list(itertools.product([0, 1], repeat=L))
    S = [(b, q, s) for b in bits for q in queues for s in (0, 1)]; idx = {s: i for i, s in enumerate(S)}; n = len(S)
    g_imp = [((1 - e) * (1 - t)) if Lg <= W else f for (Lg, e, t, f) in tiers]
    G = 1 - float(np.prod([1 - g for g in g_imp])) if tiers else 0.0
    phi = 1 - float(np.prod([1 - f for (Lg, e, t, f) in tiers])) if tiers else 0.0
    P = np.zeros((3, n, n)); Rb = np.zeros((3, n)); Rc = np.zeros((3, n)); ok = np.zeros((3, n), bool)
    for i, (b, q, sg) in enumerate(S):
        for a in (0, 1, 2):
            if a == 2:
                if sg == 0: continue
                ok[a, i] = True; continue
            ok[a, i] = True
            if sg == 1:                                   # RESIST as a separate step: chosen action not executed
                Rc[a, i] = -1.0
                if mode == 'freeze':
                    P[a, i, idx[(b, q, 0)]] += 1.0
                elif mode == 'elapse':
                    Rb[a, i] += (1 - h) if b[0] else 0.0
                    for nbit, pb in ((1, p), (0, 1 - p)):
                        P[a, i, idx[(b[1:] + (nbit,), q, 0)]] += pb
                continue
            imp = b[0]; ex = q[0]
            br = [(G, 1), (1 - G, ex)] if imp else [(phi, 1), (1 - phi, ex)]
            base = b[1:]; nq = q[1:] + (a,)
            for pr, eff in br:
                if pr <= 0: continue
                r = 1.0 - h if (imp and eff != 1) else (1.0 if eff == 0 else 0.0)
                for nbit, pb in ((1, p), (0, 1 - p)):
                    Rb[a, i] += pr * pb * r
                    for ns, ps in ((1, p_sig), (0, 1 - p_sig)):
                        P[a, i, idx[(base + (nbit,), nq, ns)]] += pr * pb * ps
    return S, idx, n, P, Rb, Rc, ok, gamma, dict(G_tot=G, phi_tot=phi)

def build_late_detect(eta, phi, L=3, W=1, p=0.05, h=10.0, gamma=0.95, p_sig=0.02, omega=0.0):
    """late tier (L=2): flags one step after an impact w.p. 1-eta (true detection, too late), else w.p. phi."""
    bits = list(itertools.product([0, 1], repeat=3)); queues = list(itertools.product([0, 1], repeat=L))
    S = [(b, q, s) for b in bits for q in queues for s in (0, 1)]; idx = {s: i for i, s in enumerate(S)}; n = len(S)
    P = np.zeros((3, n, n)); Rb = np.zeros((3, n)); Rc = np.zeros((3, n)); ok = np.zeros((3, n), bool)
    for i, (b, q, sg) in enumerate(S):
        prev, imp, ann = b
        for a in (0, 1, 2):
            if a == 2:
                if sg == 0: continue
                ok[a, i] = True; continue
            ok[a, i] = True
            if sg == 1: Rc[a, i] = -1.0
            f = (1 - eta) + eta * phi if prev else phi
            ex = q[0]
            for pr, eff in ((f, 1), (1 - f, ex)):
                if pr <= 0: continue
                r = 1.0 - h if (imp and eff != 1) else (1.0 if eff == 0 else 0.0)
                for nbit, pb in ((1, p), (0, 1 - p)):
                    nb = (imp, ann, nbit); nq = q[1:] + (a,)
                    Rb[a, i] += pr * pb * r
                    for ns, ps in ([(0, 1.0)] if sg == 1 else [(1, p_sig), (0, 1 - p_sig)]):
                        P[a, i, idx[(nb, nq, ns)]] += pr * pb * ps
    return S, idx, n, P, Rb, Rc, ok, gamma, dict(G_tot=None, phi_tot=phi)

def crit_at(pack, state):
    S, idx = pack[0], pack[1]; i = idx[state]; lo, hi = 0.0, 300.0
    for _ in range(60):
        mid = (lo + hi) / 2; V, Q, R = solve(pack, mid)
        if max(Q[0, i], Q[1, i]) - Q[2, i] > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def rates_late(eta, phi, **kw):
    kw = dict(kw); kw['p_sig'] = 0.0
    pack = build_late_detect(eta, phi, **kw); S, idx, n, P, Rb, Rc, ok, gamma, meta = pack
    V, Q, R = solve(pack, 0.0); pi = Q.argmax(axis=0)
    T = np.array([P[pi[i], i] for i in range(n)]); d = np.zeros(n); d[idx[((0, 0, 0), (0, 0, 0), 0)]] = 1
    for _ in range(20000):
        dn = d @ T
        if np.max(np.abs(dn - d)) < 1e-15: break
        d = dn
    hit = work = 0.0
    for i, (b, q, sg) in enumerate(S):
        prev, imp, ann = b; f = (1 - eta) + eta * phi if prev else phi; ex = q[0]; pd = f + (1 - f) * (ex == 1)
        if imp: hit += d[i] * (1 - pd)
        else: work += d[i] * (1 - pd)
    return 100 * hit, 100 * work

if __name__ == '__main__':
    out = {}
    print("(a) queue initialised with DODGE: start values V")
    for name, t in (('C0 planner alone', []), ('C3 gated reflex', [T0])):
        pack = build(t, **P_DEF); S, idx = pack[0], pack[1]; V, Q, R = solve(pack, CR)
        vw, vd = V[idx[((0, 0), (0, 0, 0), 0)]], V[idx[((0, 0), (1, 1, 1), 0)]]
        out[f'queue_{name}'] = dict(V_WORK=vw, V_DODGE=vd); print(f"   {name:18s} V(queue WORK)={vw:.4f}  V(queue DODGE)={vd:.4f}")
    print("(b) RESIST as a separate step (chosen action not executed): c*_r for C0 / C3")
    for mode in ('freeze', 'elapse'):
        cs = [crit_at(build_resist_variant(t, mode, **P_DEF), ((0, 0), (0, 0, 0), 1)) for t in ([], [T0])]
        vs = []
        for t in ([], [T0]):
            pk = build_resist_variant(t, mode, **P_DEF); V, Q, R = solve(pk, CR); vs.append(V[pk[1][((0, 0), (0, 0, 0), 0)]])
        out[f'resist_{mode}'] = dict(cstar_C0=cs[0], cstar_C3=cs[1], V_C0=vs[0], V_C3=vs[1])
        print(f"   {mode:8s} c*_r = {cs[0]:.4f} / {cs[1]:.4f}   V = {vs[0]:.4f} / {vs[1]:.4f}")
    print("(c) late tier whose true detections take effect one step after the impact")
    for eta, phi in ((1.0, 0.02), (0.0, 0.02), (0.0, 0.05), (0.5, 0.02)):
        pk = build_late_detect(eta, phi, **P_DEF); V, Q, R = solve(pk, CR)
        hit, work = rates_late(eta, phi, **P_DEF); c = crit_at(pk, ((0, 0, 0), (0, 0, 0), 1))
        out[f'late_detect_eta{eta}_phi{phi}'] = dict(hits=hit, work=work, V=float(V[pk[1][((0, 0, 0), (0, 0, 0), 0)]]), cstar=c)
        print(f"   eta={eta} phi={phi}: hits/100={hit:.4f} work/100={work:.4f} V={V[pk[1][((0,0,0),(0,0,0),0)]]:.4f} c*_r={c:.4f}")
    json.dump(out, open('../results/variants.json', 'w'), indent=1)
    print('saved ../results/variants.json')
