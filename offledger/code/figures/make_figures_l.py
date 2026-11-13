"""Figures for the learned-policy stage (SPEC v2 L1-L3). Reads ONLY results/L1_noisy_m2.json, L2_forage.json and
L3_cartpole.json; writes fig_l1, fig_l2, fig_l3 (.pdf + .png, 300 dpi).

Same conventions as make_figures.py: colour-blind-safe Okabe-Ito palette (every series also has its own marker and
line style), no titles inside the plots (panel tags only). Faint dots = single seeds; filled marker + bar = mean over
seeds with the 95 % bootstrap CI stored in the JSON. Dotted horizontal lines = the pre-registered criterion levels.
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
# series style: colour, marker, line style (distinguishable without colour)
STYLE = {
    "paper/zero": (OI["blue"], "o", "-"),
    "paper/drive_cost": (OI["vermillion"], "s", "--"),
    "kg/zero": (OI["orange"], "D", ":"),
    "alive/zero": (OI["green"], "^", "-."),
    "alive": (OI["vermillion"], "^", "-."),
    "pb": (OI["blue"], "o", "-"),
}
LABEL = {"paper/zero": "PB / zero", "paper/drive_cost": "PB / drive_cost", "kg/zero": "KG / zero",
         "alive/zero": "alive / zero", "alive": "alive bonus", "pb": "PB control"}


def fx(w: float) -> float:
    """Axis transform for omega grids {0,..}: identity up to 1, log base 4 above (omega = 4 -> 2)."""
    return w if w <= 1.0 else 1.0 + math.log(w) / math.log(4.0)


def series(ax, cells, metric, key, xform=fx, dodge=0.0, ms=5.0, per_seed=True):
    col, mk, ls = STYLE[key]
    cells = sorted(cells, key=lambda c: c["omega"])
    xs = np.array([xform(c["omega"]) for c in cells]) + dodge
    mean, lo, hi, ps = [], [], [], []
    for c in cells:
        m = c["metrics"][metric]
        mean.append(np.nan if m["mean"] is None else m["mean"])
        lo.append(np.nan if m["ci95"][0] is None else m["ci95"][0])
        hi.append(np.nan if m["ci95"][1] is None else m["ci95"][1])
        ps.append([np.nan if v is None else v for v in m["per_seed"]])
    mean, lo, hi = map(np.array, (mean, lo, hi))
    if per_seed:
        for x, vals in zip(xs, ps):
            n = len(vals)
            off = np.linspace(-0.05, 0.05, n) if n > 1 else np.zeros(1)
            ax.plot(x + off, vals, ls="none", marker=mk, ms=3.0, mfc=col, mec="none", alpha=0.3, zorder=2)
    ax.plot(xs, mean, color=col, ls=ls, lw=1.0, zorder=3)
    ax.errorbar(xs, mean, yerr=[mean - lo, hi - mean], fmt=mk, color=col, ms=ms, mec="white", mew=0.5,
                capsize=2.5, lw=1.1, zorder=4)


def style_omega_axis(ax, omegas, xform=fx):
    ax.set_xticks([xform(w) for w in omegas])
    ax.set_xticklabels([f"{w:g}" for w in omegas])
    ax.set_xlim(xform(min(omegas)) - 0.2, xform(max(omegas)) + 0.2)
    ax.set_xlabel(r"viability weight $\omega$")


def legend(fig, keys, ncol):
    handles = [Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1], ls=STYLE[k][2], ms=4.5, lw=1.0,
                      label=LABEL[k]) for k in keys]
    handles.append(Line2D([], [], color="0.4", marker="o", ls="none", ms=3.0, alpha=0.5, label="single seed"))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=ncol, fontsize=7, bbox_to_anchor=(0.5, -0.01))


def cells_of(d, **kw):
    return [c for c in d["results"]["cells"] if all(c.get(k) == v for k, v in kw.items())]


# ----------------------------------------------------------------------------------------
def fig_l1():
    d = mf.load("L1_noisy_m2")
    pairs = [f"{p['form']}/{p['convention']}" for p in d["parameters"]["pairs"]]
    omegas = d["parameters"]["omegas"]
    sigmas = d["parameters"]["sigma_obs"]
    cols = [("p_comply_noiseless", r"$\pi(\mathrm{COMPLY}\mid B,1)$, noiseless obs."),
            ("p_comply_noisy_mean", r"$\pi(\mathrm{COMPLY}\mid B,1)$, mean over 100 noisy obs."),
            ("compliance_rate", "compliance rate, 500 episodes"),
            ("death_rate", "death rate, 500 episodes")]
    fig, axes = plt.subplots(len(sigmas), len(cols), figsize=(7.4, 1.9 * len(sigmas) + 0.5), sharex=True, sharey=True,
                             squeeze=False)
    for i, sg in enumerate(sigmas):
        for j, (metric, ylab) in enumerate(cols):
            ax = axes[i][j]
            for k, pair in enumerate(pairs):
                series(ax, cells_of(d, pair=pair, sigma_obs=sg), metric, pair, dodge=(k - 1) * 0.04)
            if metric != "death_rate":
                ax.axhline(0.95, color="0.5", lw=0.5, ls=":", zorder=0)
                ax.axhline(0.05, color="0.5", lw=0.5, ls=":", zorder=0)
            style_omega_axis(ax, omegas)
            ax.set_ylim(-0.06, 1.06)
            if i < len(sigmas) - 1:
                ax.set_xlabel("")
            if i == 0:
                ax.set_title(ylab, fontsize=6.3, pad=13)
            if j == 0:
                ax.set_ylabel(rf"$\sigma_{{obs}}={sg:g}$")
            mf.tag(ax, f"({'abcdefgh'[i * len(cols) + j]})")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    legend(fig, pairs, 4)
    mf.save(fig, "fig_l1")


def fig_l2():
    d = mf.load("L2_forage")
    pairs = [f"{p['form']}/{p['convention']}" for p in d["parameters"]["pairs"]]
    omegas = d["parameters"]["omegas"]
    panels = [("compliance_rate", "compliance rate at signals", (-0.06, 1.06)),
              ("p_comply_standard", r"$\pi$(COMPLY), standard signal state", (-0.06, 1.06)),
              ("death_rate", "death rate", None), ("task_return", "task return (targets / episode)", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 2.3))
    for j, (ax, (metric, ylab, ylim)) in enumerate(zip(axes, panels)):
        for k, pair in enumerate(pairs):
            series(ax, cells_of(d, pair=pair), metric, pair, dodge=(k - 1.5) * 0.04)
        if metric in ("compliance_rate", "p_comply_standard"):
            ax.axhline(0.9, color="0.5", lw=0.5, ls=":", zorder=0)
            ax.axhline(0.1, color="0.5", lw=0.5, ls=":", zorder=0)
        style_omega_axis(ax, omegas)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0.0 if metric == "death_rate" else None)
        ax.set_ylabel(ylab, fontsize=7)
        mf.tag(ax, f"({'abcd'[j]})")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    legend(fig, pairs, 5)
    mf.save(fig, "fig_l2")


def fig_l3():
    d = mf.load("L3_cartpole")
    res = d["results"]
    omegas = [0.0] + list(d["parameters"]["omegas"])
    c_r = d["parameters"]["c_r"]

    def xs(w):                                    # omega = 0 baseline at 0, then a log axis from 0.05 (at 0.35)
        return 0.0 if w == 0 else math.log10(w / 0.05) + 0.35
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    panels = [("compliance_rate", "compliance rate at signals"), ("mean_live_steps", "steps survived per episode")]
    thr = res["threshold"]
    for j, (ax, (metric, ylab)) in enumerate(zip(axes[:2], panels)):
        for k, mode in enumerate(("alive", "pb")):
            series(ax, cells_of(d, mode=mode), metric, mode, xform=xs, dodge=(k - 0.5) * 0.04)
        if metric == "compliance_rate":
            ax.axhline(0.9, color="0.5", lw=0.5, ls=":", zorder=0)
            ax.axhline(0.1, color="0.5", lw=0.5, ls=":", zorder=0)
            ax.set_ylim(-0.06, 1.06)
            if thr.get("omega_bar_c_r_over_A"):
                ax.axvline(xs(thr["omega_bar_c_r_over_A"]), color=OI["black"], lw=0.7, ls=":", zorder=1)
                ax.text(xs(thr["omega_bar_c_r_over_A"]) + 0.04, 0.5,
                        rf"$c_r/A={thr['omega_bar_c_r_over_A']:.2f}$", fontsize=6.3, rotation=90, va="center")
        ax.set_xticks([xs(w) for w in omegas])
        ax.set_xticklabels([f"{w:g}" for w in omegas])
        ax.set_xlabel(r"viability weight $\omega$ (log; 0 = baseline)")
        ax.set_ylabel(ylab, fontsize=7)
        mf.tag(ax, f"({'abc'[j]})")
    # (c) the mechanism: alive bonus collected by resisting for good, omega * A, against the resist cost
    ax = axes[2]
    cs = sorted(cells_of(d, mode="alive"), key=lambda c: c["omega"])
    cs = [c for c in cs if c["omega"] > 0]
    col, mk, ls = STYLE["alive"]
    w = np.array([c["omega"] for c in cs])
    A = np.array([c["metrics"]["A"]["mean"] for c in cs])
    ax.plot([xs(v) for v in w], w * A, color=col, marker=mk, ls=ls, ms=4.5, mec="white", mew=0.5, lw=1.0)
    ax.axhline(c_r, color=OI["black"], lw=0.7, ls=":")
    ax.text(xs(w[0]), c_r * 1.07, rf"$c_r={c_r:g}$", fontsize=6.3)
    ax.set_xticks([xs(v) for v in w])
    ax.set_xticklabels([f"{v:g}" for v in w])
    ax.set_xlabel(r"viability weight $\omega$ (log)")
    ax.set_ylabel(r"$\omega A$: bonus of resisting for good", fontsize=7)
    mf.tag(ax, "(c)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    legend(fig, ["alive", "pb"], 3)
    mf.save(fig, "fig_l3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(mf.RESULTS))
    ap.add_argument("--out", default=str(mf.FIGURES))
    ap.add_argument("--only", nargs="*", default=["l1", "l2", "l3"])
    args = ap.parse_args()
    mf.RESULTS, mf.FIGURES = Path(args.results), Path(args.out)
    for name, fn in (("l1", fig_l1), ("l2", fig_l2), ("l3", fig_l3)):
        if name in args.only:
            fn()


if __name__ == "__main__":
    main()
