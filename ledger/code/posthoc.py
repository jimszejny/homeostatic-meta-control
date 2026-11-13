"""POST-HOC closed-form evaluations quoted in the text (not pre-specified; not bisection-verified
except where stated). Writes results/posthoc.json."""
import json, os
import numpy as np
from chain import closed_cstar, CR
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
g = 0.95
res = {}
# resist threshold omega-bar = (c_r - c*(0))/beta for death@1, off L1, delta = 0.2
for name, G, phi in [('planner', 0.0, 0.0), ('T1 G=0.50', 0.5, 0.02), ('T1 G=0.80', 0.8, 0.02)]:
    c, c0, b, K, M = closed_cstar(Gamma=G, phi=phi, delta=0.2, off_price=g, death_price=g)
    res[f'omega_bar_L1_death1_d0.2_{name}'] = (CR - c0) / b
# relative change of c* from omega 0 to 4, planner, delta 0.2, death@1 off L0
c4 = closed_cstar(delta=0.2, omega=4.0, death_price=g)[0]; c0 = closed_cstar(delta=0.2)[0]
res['rel_change_planner_d0.2_death1_offL0_omega4'] = c4 / c0 - 1
# min combined beta (off L1 + death@1) over (Gamma, delta) in [0,1]^2, phi = 0
res['min_beta_L1_death1_square_phi0'] = min(closed_cstar(Gamma=G, delta=d, off_price=g, death_price=g)[2]
                                            for G in np.linspace(0, 1, 101) for d in np.linspace(0, 1, 101))
# omega at which L4 c* reaches 0 (planner)
res['omega_L4_zero_planner'] = closed_cstar()[0] / ((1 - g) * 0.05 * (0.5 + g + 12.818115942028983))
json.dump(res, open(os.path.join(OUT, 'posthoc.json'), 'w'), indent=1)
for k, v in res.items(): print(k, v)
