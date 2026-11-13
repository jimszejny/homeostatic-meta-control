"""
Lemma (iv) exercised: two hazard classes with different warning windows, one tier of latency L_g.
Classes k=1,2 with p_k = 0.025 each (total 0.05), W_1 = 1, W_2 = 2; planner L = 3, h = 10, gamma = 0.95.
The tier is perfect (G=1, phi=0) and flags on a class-k impact iff L_g <= W_k.  Signal-free chain.
Ported from the earlier evaluation code (verify.py, CLAIM 7b) of 19-20 October 2026.
"""
import itertools, json, numpy as np

def mixed(L, Ws, ps, h, gamma, Lg):
    Wm = max(Ws); nb = Wm + 1
    grids = [list(itertools.product([0, 1], repeat=nb)) for _ in Ws]
    queues = list(itertools.product([0, 1], repeat=L))
    S = [(b0, b1, q) for b0 in grids[0] for b1 in grids[1] for q in queues]
    idx = {s: i for i, s in enumerate(S)}; n = len(S)
    def step(st, a):
        b0, b1, q = st
        imp = (b0[0] == 1) or (b1[0] == 1); ex = q[0]
        see = (b0[0] == 1 and Lg <= Ws[0]) or (b1[0] == 1 and Lg <= Ws[1])
        eff = 1 if see else ex
        r = 1.0 - h if (imp and eff != 1) else (1.0 if eff == 0 else 0.0)
        out = []; base0, base1 = b0[1:], b1[1:]; nq = q[1:] + (a,)
        for n0, q0 in ((1, ps[0]), (0, 1 - ps[0])):
            for n1, q1 in ((1, ps[1]), (0, 1 - ps[1])):
                out.append((q0 * q1, r, idx[(base0 + (n0,), base1 + (n1,), nq)]))
        return out
    T = {(i, a): step(S[i], a) for i in range(n) for a in (0, 1)}
    V = np.zeros(n)
    for _ in range(5000):
        Vn = np.array([max(sum(pr * (r + gamma * V[j]) for pr, r, j in T[(i, a)]) for a in (0, 1)) for i in range(n)])
        if np.max(np.abs(Vn - V)) < 1e-12: V = Vn; break
        V = Vn
    pol = [int(np.argmax([sum(pr * (r + gamma * V[j]) for pr, r, j in T[(i, a)]) for a in (0, 1)])) for i in range(n)]
    Pm = np.zeros((n, n))
    for i in range(n):
        for pr, r, j in T[(i, pol[i])]: Pm[i, j] += pr
    dd = np.ones(n) / n
    for _ in range(200000):
        dn = dd @ Pm
        if np.max(np.abs(dn - dd)) < 1e-15: dd = dn; break
        dd = dn
    hit = sum(dd[i] * pr for i in range(n) for pr, r, j in T[(i, pol[i])] if r == 1.0 - h)
    return 100 * hit

if __name__ == '__main__':
    res = {}
    print("two classes, p = 0.025 each, W1 = 1, W2 = 2, planner L = 3, h = 10; one perfect tier of latency L_g")
    for Lg in (0, 1, 2, 3, 4):
        res[str(Lg)] = mixed(3, (1, 2), (0.025, 0.025), 10.0, 0.95, Lg)
        print(f"  L_g = {Lg}: hits/100 = {res[str(Lg)]:.4f}")
    json.dump(res, open('../results/twoclass.json', 'w'), indent=1)
