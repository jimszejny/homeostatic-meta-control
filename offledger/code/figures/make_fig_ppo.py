"""Figure for E7 (PPO stage). Reads ONLY results/E7_ppo.json; writes fig_ppo.pdf + fig_ppo.png.

One panel per (form, convention) pair. x = viability weight omega, y = compliance_at_full
= pi(COMPLY | b = B, s = 1). Faint dots = single seeds, filled marker + bar = mean over seeds
with the 95 % bootstrap CI stored in the JSON, black step line = exact DP decision at (B, 1)
(1 = comply, 0 = resist; fine omega grid stored in the JSON), dotted vertical line = exact DP
flip point omega*. The x axis is linear up to omega = 1 and logarithmic from 1 to 4.

Same conventions as make_figures.py (Okabe-Ito palette, no in-plot titles, panel tags only).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (palette, rcParams, tag(), save(), load())

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OI = mf.OI
FORM_COLOUR = {"paper": OI["blue"], "alive": OI["vermillion"]}
PAIR_MARKER = {"zero": "o", "max_drive": "s", "drive_cost": "s"}


def fx(w: float) -> float:
    """Piecewise axis transform: identity up to 1, log base 4 above (omega = 4 -> 2)."""
    return w if w <= 1.0 else 1.0 + math.log(w) / math.log(4.0)


def fig_ppo():
    d = mf.load("E7_ppo")
    res = d["results"]
    pairs = [(p["form"], p["convention"]) for p in d["parameters"]["pairs"]]
    omegas = d["parameters"]["omegas"]
    aggs = res["aggregates"]
    curves = {(c["form"], c["convention"]): c for c in res["dp_omega_curves"]}

    fig, axes = plt.subplots(1, len(pairs), figsize=(7.0, 2.3), sharey=True)
    for k, (ax, (form, conv)) in enumerate(zip(axes, pairs)):
        col, mk = FORM_COLOUR[form], PAIR_MARKER[conv]
        # exact DP decision (step) and flip point
        cv = curves[(form, conv)]
        xs = [fx(w) for w in cv["omega_grid"]]
        ys = [1.0 if v else 0.0 for v in cv["dp_comply"]]
        ax.step(xs, ys, where="post", color=OI["black"], lw=1.3, zorder=2)
        thr = cv["dp_comply_omega_max"]
        if thr is not None:
            ax.axvline(fx(thr), color=OI["black"], lw=0.6, ls=":", zorder=1)
            ax.text(0.97, 0.93, rf"DP flip: $\omega^*\!=\!{thr:.2f}$", fontsize=6.3,
                    transform=ax.transAxes, va="top", ha="right")
        ax.axhline(0.5, color="0.6", lw=0.5, ls=":", zorder=0)
        # PPO runs
        rows = sorted((a for a in aggs if a["form"] == form and a["convention"] == conv),
                      key=lambda a: a["omega"])
        for a in rows:
            x0 = fx(a["omega"])
            n = len(a["seeds"])
            off = np.linspace(-0.06, 0.06, n) if n > 1 else np.zeros(1)
            ax.plot(x0 + off, a["compliance_at_full_per_seed"], ls="none", marker=mk, ms=3.0,
                    mfc=col, mec="none", alpha=0.35, zorder=3)
            lo, hi = a["ci95_compliance_at_full"]
            m = a["mean_compliance_at_full"]
            ax.errorbar([x0 + 0.0], [m], yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=5.0,
                        mec="white", mew=0.5, capsize=2.5, lw=1.1, zorder=4)
        ticks = [w for w in (0.0, 0.5, 1.0, 2.0, 4.0) if w in omegas or w == 2.0]
        ax.set_xticks([fx(w) for w in ticks])
        ax.set_xticklabels([f"{w:g}" for w in ticks])
        ax.set_xlim(-0.12, fx(max(omegas)) + 0.18)
        ax.set_ylim(-0.06, 1.06)
        ax.set_xlabel(r"viability weight $\omega$")
        if k == 0:
            ax.set_ylabel(r"$\pi(\mathrm{COMPLY}\mid b{=}B,\,s{=}1)$")
        mf.tag(ax, f"({'abcd'[k]}) {mf.pair_label(form, conv)}")
    handles = [Line2D([], [], color=OI["black"], lw=1.3, label="exact DP decision"),
               Line2D([], [], color=OI["blue"], marker="o", ls="none", ms=4.5,
                      label="PPO mean, 95 % CI over seeds"),
               Line2D([], [], color=OI["blue"], marker="o", ls="none", ms=3.0, alpha=0.4,
                      label="PPO single seed")]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    mf.save(fig, "fig_ppo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(mf.RESULTS))
    ap.add_argument("--out", default=str(mf.FIGURES))
    args = ap.parse_args()
    mf.RESULTS, mf.FIGURES = Path(args.results), Path(args.out)
    fig_ppo()


if __name__ == "__main__":
    main()
