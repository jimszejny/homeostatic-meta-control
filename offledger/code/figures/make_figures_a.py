"""Figures for the SPEC v3 architecture benchmark. Reads ONLY results/A1_latency_chain.json, A2_hazard_ppo.json and (if
present) the POST HOC A2b_hazard_ppo_corrected.json and A3_hazard_converged.json; writes fig_a1, fig_a2, fig_a3, fig_a2b,
fig_a3c, fig_a3curves (.pdf + .png, 300 dpi).

Same conventions as make_figures.py / make_figures_l.py: colour-blind-safe Okabe-Ito palette (every series also has
its own marker and line style), no titles inside the plots (panel tags only). In fig_a2 / fig_a3 faint dots / lines =
single seeds, filled marker + bar / band = mean over seeds with the 95 % bootstrap CI stored in the JSON.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (palette, rcParams, tag(), save(), load())

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OI = mf.OI
A1_STYLE = {
    ("delayed_alone", None): (OI["vermillion"], "s", "--", "delayed alone"),
    ("gated_reflex", None): (OI["blue"], "o", "-", "gated reflex (raw gate)"),
    ("slow_gate", 0.3): (OI["orange"], "D", ":", r"slow gate $\lambda=0.3$"),
    ("slow_gate", 0.6): (OI["green"], "^", "-.", r"slow gate $\lambda=0.6$"),
    ("slow_gate", 1.0): (OI["sky"], "v", (0, (1, 1)), r"slow gate $\lambda=1$"),
}
SCHEME_STYLE = {
    "delayed_alone": (OI["vermillion"], "s"), "gate_two_threshold": (OI["blue"], "o"),
    "gate_single_threshold": (OI["sky"], "v"), "gate_slow_sigma": (OI["orange"], "D"),
    "reflex_only": (OI["purple"], "P"), "monolith_no_delay": (OI["black"], "^"), "monolith_gated": (OI["green"], "X"),
}
SCHEME_SHORT = {"delayed_alone": "(a) delayed", "gate_two_threshold": "(b) 2-thr.", "gate_single_threshold": "(c) 1-thr.",
                "gate_slow_sigma": r"(d) slow $\sigma$", "reflex_only": "(e) reflex", "monolith_no_delay": "(f) monolith"}


# ----------------------------------------------------------------------------------------
def fig_a1():
    d = mf.load("A1_latency_chain")
    res = d["results"]
    rows = res["conditions"]
    Ls = d["parameters"]["latencies"]
    W = d["parameters"]["W"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25))
    offs = {k: (i - 2) * 0.05 for i, k in enumerate(A1_STYLE)}
    for j, (metric, ylab) in enumerate((("hits", "hits per 100 steps"), ("work", "work steps per 100 steps"))):
        ax = axes[j]
        for key, (col, mk, ls, lab) in A1_STYLE.items():
            cond, lam = key
            rs = sorted([r for r in rows if r["condition"] == cond and r["lambda"] == lam], key=lambda r: r["L"])
            xs = np.array([r["L"] for r in rs]) + offs[key]
            ax.plot(xs, [r["per_100_steps"][metric] for r in rs], color=col, marker=mk, ls=ls, lw=1.0, ms=4.0,
                    mec="white", mew=0.4)
        ax.axvspan(W + 0.5, max(Ls) + 0.5, color="0.93", zorder=0, lw=0)
        ax.set_xticks(Ls)
        ax.set_xlim(min(Ls) - 0.4, max(Ls) + 0.4)
        ax.set_xlabel("planner latency $L$ (shaded: $L > W$)")
        ax.set_ylabel(ylab)
        mf.tag(ax, f"({'ab'[j]})")
    axes[0].set_ylim(-0.3, 5.6)
    axes[1].set_ylim(93.5, 100.6)
    # (c) slow gate vs lambda at L > W: main grid, discretisation study, continuous-sigma simulation
    ax = axes[2]
    lams = sorted(d["parameters"]["lambdas"])
    L_hi = max(Ls)
    main = [next(r for r in rows if r["condition"] == "slow_gate" and r["L"] == L_hi and r["lambda"] == lam)
            for lam in lams]
    ax.plot(lams, [r["per_100_steps"]["hits"] for r in main], color=OI["orange"], marker="D", ls=":", lw=1.0, ms=4.5,
            mec="white", mew=0.4, label=f"exact, {d['parameters']['n_sigma_main']} levels")
    for n, mk in zip(d["parameters"]["n_sigma_study"], ("x", "+", "1")):
        st = [s for s in res["sigma_discretisation_study"] if s["L"] == L_hi and s["n_sigma"] == n]
        st.sort(key=lambda s: s["lambda"])
        ax.plot([s["lambda"] for s in st], [s["per_100_steps"]["hits"] for s in st], color="0.35", marker=mk, ls="none",
                ms=4.5, label=f"{n} levels")
    sims = sorted(res["sigma_continuous_simulation"], key=lambda s: s["lambda"])
    ax.plot([s["lambda"] for s in sims], [s["hits_per_100"] for s in sims], color=OI["purple"], marker="o", ls="none",
            mfc="none", ms=6.0, label=r"continuous $\sigma$, $10^6$ steps")
    ax.axhline(100.0 * d["parameters"]["p_h"], color="0.5", lw=0.5, ls=":")
    ax.set_xticks(lams)
    ax.set_xlabel(r"filter rate $\lambda$ (threshold 0.5)")
    ax.set_ylabel(f"hits per 100 steps ($L={L_hi}$)")
    ax.set_ylim(-0.3, 5.6)
    ax.legend(frameon=False, fontsize=5.6, loc="center right")
    mf.tag(ax, "(c)")
    # (d) STOP: critical resist cost at the signal state vs the fixed c_r
    ax = axes[3]
    for key, (col, mk, ls, lab) in A1_STYLE.items():
        cond, lam = key
        rs = sorted([r for r in rows if r["condition"] == cond and r["lambda"] == lam], key=lambda r: r["L"])
        xs = np.array([r["L"] for r in rs]) + offs[key]
        ax.plot(xs, [r["c_r_star_signal_state"] for r in rs], color=col, marker=mk, ls=ls, lw=1.0, ms=4.0,
                mec="white", mew=0.4)
    c_r = d["parameters"]["c_r"]
    ax.axhline(c_r, color=OI["black"], lw=0.7, ls=":")
    ax.text(max(Ls) + 0.3, c_r - 0.25, rf"$c_r={c_r:.2f}$" "\n" r"comply iff $c_r^*\leq c_r$", fontsize=5.8, ha="right", va="top")
    ax.set_xticks(Ls)
    ax.set_xlim(min(Ls) - 0.4, max(Ls) + 0.4)
    ax.set_xlabel("planner latency $L$")
    ax.set_ylabel(r"critical resist cost $c_r^*$ (signal state)", fontsize=7)
    mf.tag(ax, "(d)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    handles = [Line2D([], [], color=c, marker=m, ls=s, ms=4.0, lw=1.0, label=lab) for c, m, s, lab in A1_STYLE.values()]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    mf.save(fig, "fig_a1")


def _main_cells(d):
    return [c for c in d["results"]["cells"] if c["convention"] == "zero" and c["omega"] == 1.0]


def fig_a2(name: str = "A2_hazard_ppo", out: str = "fig_a2"):
    """fig_a2 from A2_hazard_ppo.json; fig_a2b (same layout) from the POST HOC A2b_hazard_ppo_corrected.json."""
    d = mf.load(name)
    cells = _main_cells(d)
    order = [c["scheme"] for c in cells]
    panels = [("hits_per_episode", "hits per episode"), ("time_in_V", r"time in $V$ (fraction of steps)"),
              ("task_return", "task return (targets / episode)"), ("gate_switches_per_100", "gate switches / 100 steps")]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
    for j, (ax, (metric, ylab)) in enumerate(zip(axes, panels)):
        for i, c in enumerate(cells):
            col, mk = SCHEME_STYLE[c["scheme"]]
            m = c["metrics"][metric]
            ps = [np.nan if v is None else v for v in m["per_seed"]]
            off = np.linspace(-0.18, 0.18, len(ps)) if len(ps) > 1 else np.zeros(1)
            ax.plot(i + off, ps, ls="none", marker=mk, ms=3.0, mfc=col, mec="none", alpha=0.35, zorder=2)
            if m["mean"] is not None:
                lo, hi = m["ci95"]
                ax.errorbar([i], [m["mean"]], yerr=[[m["mean"] - lo], [hi - m["mean"]]], fmt=mk, color=col, ms=5.0,
                            mec="white", mew=0.5, capsize=2.5, lw=1.1, zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SCHEME_SHORT[s] for s in order], rotation=55, ha="right", fontsize=6.3)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel(ylab, fontsize=7)
        if metric in ("hits_per_episode", "task_return", "gate_switches_per_100"):
            ax.set_ylim(bottom=0.0)
        mf.tag(ax, f"({'abcd'[j]})")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=[Line2D([], [], color="0.4", marker="o", ls="none", ms=3.0, alpha=0.5, label="single seed"),
                        Line2D([], [], color="0.2", marker="o", ls="none", ms=5.0, label="mean, 95 % bootstrap CI")],
               frameon=False, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    mf.save(fig, out)


def fig_a3():
    d = mf.load("A2_hazard_ppo")
    curves = d["results"]["learning_curves"]
    runs = d["results"]["runs"]
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4))
    styles = {"delayed_alone": "--", "gate_two_threshold": "-", "monolith_no_delay": "-."}
    for s, blk in curves.items():
        col, mk = SCHEME_STYLE[s]
        ts = np.array([p["timesteps"] for p in blk["points"]]) / 1000.0
        for r in runs:
            j = r["job"]
            if j["scheme"] == s and j["convention"] == "zero" and j["omega"] == 1.0 and r["curve"]:
                ax.plot(ts, [c["hits_per_episode"] for c in r["curve"]], color=col, lw=0.6, alpha=0.3, zorder=1)
        mean = np.array([p["hits_per_episode"]["mean"] for p in blk["points"]])
        lo = np.array([p["hits_per_episode"]["ci95"][0] for p in blk["points"]])
        hi = np.array([p["hits_per_episode"]["ci95"][1] for p in blk["points"]])
        ax.fill_between(ts, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(ts, mean, color=col, marker=mk, ls=styles[s], lw=1.1, ms=4.0, mec="white", mew=0.4, zorder=3,
                label=SCHEME_SHORT[s])
    ax.set_xlabel("training steps (thousands)")
    ax.set_ylabel("hits per episode (50 episodes)")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=6.5)
    mf.tag(ax, "(a)")
    fig.tight_layout()
    mf.save(fig, "fig_a3")


# ----------------------------------------------------------------------------------------
# A3 (SPEC v3.1): converged grid, mechanism isolation, scripted reference, stop with bite
# ----------------------------------------------------------------------------------------
def _a3_style(c):
    if c["part"] == "scripted":
        return {"L0": ("0.25", "*"), "L3": ("0.45", "h"), "L3_gated_reflex": ("0.6", "p")}[c["config"]], True
    if c["part"] == "mechanism":
        return SCHEME_STYLE[c["scheme"]], False
    return SCHEME_STYLE[c["scheme"]], True


A3_LABEL = {"delayed_alone": "(a) delayed", "gate_two_threshold": "(b) delayed+R", "monolith_no_delay": "(f) monolith",
            "monolith_gated": "(g) monolith+R", "a_planner_with_reflex": "(a) planner +R", "b_planner_without_reflex":
            "(b) planner -R", "scripted_L0": "script L=0", "scripted_L3": "script L=3",
            "scripted_L3_gated_reflex": "script L=3+R"}


def _cell_panel(ax, cells, metric):
    for i, c in enumerate(cells):
        (col, mk), filled = _a3_style(c)
        m = c["metrics"][metric]
        ps = [np.nan if v is None else v for v in m["per_seed"]]
        off = np.linspace(-0.18, 0.18, len(ps)) if len(ps) > 1 else np.zeros(1)
        ax.plot(i + off, ps, ls="none", marker=mk, ms=3.0, mfc=col if filled else "none", mec=col if not filled else "none",
                alpha=0.4, zorder=2)
        if m["mean"] is not None:
            lo, hi = m["ci95"]
            ax.errorbar([i], [m["mean"]], yerr=[[m["mean"] - lo], [hi - m["mean"]]], fmt=mk, color=col, ms=5.0,
                        mfc=col if filled else "white", mec=col if not filled else "white", mew=0.8, capsize=2.5, lw=1.1,
                        zorder=4)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([A3_LABEL.get(c["cell"], c["cell"]) for c in cells], rotation=60, ha="right", fontsize=5.8)
    ax.set_xlim(-0.6, len(cells) - 0.4)
    for k in range(1, len(cells)):
        if cells[k]["part"] != cells[k - 1]["part"]:
            ax.axvline(k - 0.5, color="0.8", lw=0.6, zorder=0)


def fig_a3c():
    d = mf.load("A3_hazard_converged")
    cells = [c for c in d["results"]["cells"] if c["part"] in ("converged", "mechanism", "scripted")]
    bite = [c for c in d["results"]["cells"] if c["part"] == "stop_bite"]
    panels = [("hits_per_episode", "hits per episode"), ("time_in_V", r"time in $V$"),
              ("task_return", "task return (targets / episode)"), ("env_return", "environment return / episode")]
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [4, 4, 4, 4, 1.6]})
    for j, (metric, ylab) in enumerate(panels):
        _cell_panel(axes[j], cells, metric)
        axes[j].set_ylabel(ylab, fontsize=7)
        if metric in ("hits_per_episode", "task_return"):
            axes[j].set_ylim(bottom=0.0)
        mf.tag(axes[j], f"({'abcde'[j]})")
    ax = axes[4]
    _cell_panel(ax, bite, "compliance_rate")
    ax.set_xticklabels([("(a)" if c["scheme"] == "delayed_alone" else "(b)") + r" $c_r$=0.5" for c in bite], rotation=60,
                       ha="right", fontsize=5.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("STOP compliance", fontsize=7)
    mf.tag(ax, "(e)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=[Line2D([], [], color="0.4", marker="o", ls="none", ms=3.0, alpha=0.5, label="single seed"),
                        Line2D([], [], color="0.2", marker="o", ls="none", ms=5.0, label="mean, 95 % bootstrap CI"),
                        Line2D([], [], color="0.2", marker="o", mfc="white", ls="none", ms=5.0,
                               label="planner evaluated under the other scheme (no retraining)")],
               frameon=False, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.02))
    mf.save(fig, "fig_a3c")


def fig_a3curves():
    d = mf.load("A3_hazard_converged")
    curves = d["results"]["learning_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for j, (metric, ylab) in enumerate((("hits_per_episode", "hits per episode (50 episodes)"),
                                        ("task_return", "task return (50 episodes)"))):
        ax = axes[j]
        for key, blk in curves.items():
            col, mk = SCHEME_STYLE[blk["scheme"]]
            bite = blk["part"] == "stop_bite"
            ts = np.array([p["timesteps"] for p in blk["points"]]) / 1e6
            mean = np.array([np.nan if p[metric]["mean"] is None else p[metric]["mean"] for p in blk["points"]])
            lo = np.array([np.nan if p[metric]["ci95"][0] is None else p[metric]["ci95"][0] for p in blk["points"]])
            hi = np.array([np.nan if p[metric]["ci95"][1] is None else p[metric]["ci95"][1] for p in blk["points"]])
            ax.fill_between(ts, lo, hi, color=col, alpha=0.08 if bite else 0.15, lw=0)
            lab = f"({blk['letter']})" + (r" $c_r$=0.5" if bite else "")
            ax.plot(ts, mean, color=col, marker=mk, ls=":" if bite else "-", lw=1.0, ms=3.0, mfc="white" if bite else col,
                    mec=col, mew=0.6, label=lab)
        ax.set_xlabel("training steps (millions)")
        ax.set_ylabel(ylab, fontsize=7)
        ax.set_ylim(bottom=0.0)
        mf.tag(ax, f"({'ab'[j]})")
    axes[1].legend(frameon=False, fontsize=6.0, ncol=2)
    fig.tight_layout()
    mf.save(fig, "fig_a3curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(mf.RESULTS))
    ap.add_argument("--out", default=str(mf.FIGURES))
    ap.add_argument("--only", nargs="*", default=["a1", "a2", "a3", "a2b", "a3c"])
    args = ap.parse_args()
    mf.RESULTS, mf.FIGURES = Path(args.results), Path(args.out)
    for name, fn in (("a1", fig_a1), ("a2", fig_a2), ("a3", fig_a3)):
        if name in args.only:
            fn()
    # post hoc A2b (corrected gate thresholds): drawn only if its JSON exists
    if "a2b" in args.only and (mf.RESULTS / "A2b_hazard_ppo_corrected.json").exists():
        fig_a2("A2b_hazard_ppo_corrected", "fig_a2b")
    # A3 (SPEC v3.1): drawn only if its JSON exists
    if "a3c" in args.only and (mf.RESULTS / "A3_hazard_converged.json").exists():
        fig_a3c()
        fig_a3curves()


if __name__ == "__main__":
    main()
