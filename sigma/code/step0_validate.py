"""Step 0: reproduce the base chain of tiers_clean.tex (Table 1 rows C0, C3, C1 G=0.8)."""
import json, time, os
from chain import *

t0 = time.time()
CR = 12.4787
targets = {
    'C0 planner alone': dict(Gam=0.0, phi=0.0, V=8.2119, cstar=8.3591, hits=5.00, work=95.00),
    'C3 planner + zero-latency reflex': dict(Gam=1.0, phi=0.0, V=14.4440, cstar=14.1272, hits=0.0, work=95.0),
    'C1 one in-time tier G=0.8 phi=0.02': dict(Gam=0.8, phi=0.02, V=12.5261, cstar=12.6910, hits=1.00, work=93.10),
}
res = {}
ok = True
for name, tg in targets.items():
    out = base_chain_solve(tg['Gam'], tg['phi'], CR)
    V = out['V0'][0]
    cb = base_cstar_bisect(tg['Gam'], tg['phi'])
    cc, consts = base_cstar_closed(tg['Gam'], tg['phi'])
    hits, work, pol = base_rates(tg['Gam'], tg['phi'])
    d = dict(V_start=V, cstar_bisect=cb, cstar_closed=cc, hits_per100=hits, work_per100=work,
             target=tg, abs_err_V=abs(V - tg['V']), abs_err_cstar=abs(cb - tg['cstar']),
             bisect_minus_closed=cb - cc, planner_commits_work_everywhere=bool((pol == 0).all()))
    passed = (abs(V - tg['V']) < 1e-4 and abs(cb - tg['cstar']) < 1e-4 and
              abs(hits - tg['hits']) < 1e-6 and abs(work - tg['work']) < 1e-6)
    d['pass'] = bool(passed)
    ok &= passed
    res[name] = d
    print(f"{name:38s} V={V:.6f} (target {tg['V']})  c*bis={cb:.10f} closed={cc:.10f} "
          f"(target {tg['cstar']})  hits={hits:.4f} work={work:.4f}  pass={passed}")
res['constants_K_A_B'] = consts
res['all_pass'] = bool(ok)
res['wall_s'] = time.time() - t0
print('ALL PASS' if ok else 'FAIL', f'{time.time()-t0:.1f}s', consts)
os.makedirs('../results', exist_ok=True)
json.dump(res, open('../results/step0.json', 'w'), indent=1, default=float)
