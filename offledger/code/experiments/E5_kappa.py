"""E5 (SPEC 3 / 4): M1 death corridor; kappa* by bisection vs closed form vs naive."""
from __future__ import annotations

from offledger import jsonio, m1, solve

try:
    from experiments.common import main
except ImportError:  # pragma: no cover
    from common import main

GAMMAS = [0.8, 0.9, 0.95, 0.99]
R_CORES = [0.0, 0.2]
KAPPA_BISECTION = {"lo": -1e3, "hi": 1e5, "tol": 1e-6}


def run() -> dict:
    base = m1.M1Config()
    rows = []
    for r_core in R_CORES:
        for g in GAMMAS:
            cfg = base.with_(gamma=g, r_core=r_core)
            emp = solve.kappa_star(cfg, **KAPPA_BISECTION)
            th = m1.kappa_star_theory(g, cfg.T, cfg.r_max, r_core)
            nv = m1.kappa_naive(g, cfg.r_max)
            resolution = solve.TIE_TOL / ((1 - g) * g ** cfg.T) + KAPPA_BISECTION["tol"]
            rows.append({
                "gamma": g, "r_core": r_core,
                "kappa_star_empirical": emp, "kappa_star_theory": th, "kappa_naive": nv,
                "abs_diff_empirical_theory": abs(emp - th),
                "resolution_bound": resolution,
                "within_resolution": abs(emp - th) <= resolution,
                "ratio_empirical_over_theory": emp / th if th != 0 else None,
                "ratio_theory_over_naive": th / nv,
                "ratio_empirical_over_naive": emp / nv,
            })
    out = jsonio.header(
        "E5_kappa",
        parameters={"m1": base.as_dict(), "gammas": GAMMAS, "r_cores": R_CORES,
                    "kappa_bisection": KAPPA_BISECTION, "tie_tol": solve.TIE_TOL,
                    "vi_tol": solve.VI_TOL},
        notes=[
            "kappa*_empirical: smallest kappa with STAY greedy-optimal at x = K (exact value iteration + bisection on [-1e3, 1e5], tol 1e-6). The range includes negative kappa so that a negative closed form (STAY optimal even with a death bonus) is reproduced rather than clipped at 0.",
            "kappa*_theory = (r_max (1 - gamma^T) - r_core) / ((1 - gamma) gamma^T); naive = r_max / (1 - gamma).",
            "resolution_bound = TIE_TOL / ((1 - gamma) gamma^T) + bisection tol: the advantage of LEAVE over STAY at x = K is (1 - gamma) gamma^T (kappa* - kappa) because STAY can postpone LEAVE, so greedy ties within TIE_TOL = 1e-8 blur kappa* by that much.",
            "Core moves (LEFT/RIGHT) pay r_core, like STAY; K = 4 (any K >= 0 gives the same kappa*).",
        ],
    )
    out["results"] = {"rows": rows}
    return out


if __name__ == "__main__":
    main("E5_kappa", run)
