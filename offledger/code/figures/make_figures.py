"""Figures (SPEC section 5). Reads ONLY the JSON files in results/; writes PDF + PNG (300 dpi).

Colour-blind-safe Okabe-Ito palette; no titles inside the plots (panels carry a short tag).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("OFFLEDGER_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("OFFLEDGER_FIGURES", ROOT / "figures"))

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
FORM_STYLE = {"paper": (OI["blue"], "o", "-"), "kg": (OI["orange"], "s", "--"),
              "alive": (OI["green"], "^", "-.")}
# v1.1 label conventions: PB / KG / alive and zero / drive_cost / kappa (max_drive = drive_cost)
FORM_LABEL = {"paper": "PB", "kg": "KG", "alive": "alive"}
CONV_LABEL = {"zero": "zero", "max_drive": "drive_cost", "drive_cost": "drive_cost", "kappa": "kappa"}


def pair_label(form, conv):
    return f"{FORM_LABEL.get(form, form)} / {CONV_LABEL.get(conv, conv)}"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def tag(ax, text):
    """Panel tag above the axes frame (not a plot title)."""
    ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            fontweight="bold")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.pdf", FIGURES / f"{name}.png")


def fig_ledger():
    d = load("E1_ledger")
    res = d["results"]
    convs = d["parameters"]["conventions"]
    fig, axes = plt.subplots(1, len(convs), figsize=(7.0, 2.3), sharey=True)
    for k, (ax, conv) in enumerate(zip(axes, convs)):
        for c in res["configs"]:
            if c["convention"] != conv:
                continue
            col, mk, ls = FORM_STYLE[c["form"]]
            om = [r["omega"] for r in c["rows"] if r["c_r_star_B"] is not None]
            cs = [r["c_r_star_B"] for r in c["rows"] if r["c_r_star_B"] is not None]
            ax.plot(om, cs, color=col, marker=mk, ls=ls, ms=3.5, lw=1.2, label=FORM_LABEL[c["form"]])
        ax.axhline(res["c_task_star"], color=OI["black"], lw=0.7, ls=":", label=r"task-only $c_r^*$")
        ax.set_xlabel(r"viability weight $\omega$")
        if k == 0:
            ax.set_ylabel(r"critical resist cost $c_r^*$ at $(B,1)$")
            ax.legend(frameon=False, loc="center left")
        tag(ax, f"({'abc'[k]}) OFF ledger: {CONV_LABEL[conv]}")
    save(fig, "fig_ledger")


def fig_osg():
    """v1.1 E4 (DEFER / DISABLE): omega_bar(rho) at c_r = 12.479 per form / convention; hollow
    markers = omega_bar_pred; 'none' (DEFER up to omega = 64) drawn as an up-arrow at the top."""
    d = load("E4_osg")
    res = d["results"]
    rhos = d["parameters"]["rhos"]
    forms, convs = d["parameters"]["forms"], d["parameters"]["conventions"]
    c_r = max(d["parameters"]["c_rs"])
    blocks = {(b["form"], b["convention"]): b for b in res["blocks"] if b["c_r"] == c_r}
    y_none, y_lo, y_hi = 150.0, 0.1, 300.0
    fig, axes = plt.subplots(len(convs), len(forms), figsize=(7.0, 4.0), sharex=True, sharey=True)
    for i, conv in enumerate(convs):
        for j, form in enumerate(forms):
            ax = axes[i, j]
            col = FORM_STYLE[form][0]
            rows = blocks[(form, conv)]["rows"]
            for key, filled in (("omega_bar", True), ("omega_bar_pred", False)):
                xs_num = [r["rho"] for r in rows if isinstance(r[key], (int, float))]
                ys_num = [max(r[key], y_lo * 1.2) for r in rows if isinstance(r[key], (int, float))]
                xs_none = [r["rho"] for r in rows if r[key] == "none"]
                kw = dict(color=col, ls="-" if filled else "none", lw=1.1 if filled else 0,
                          marker="o", ms=4.0 if filled else 7.0,
                          mfc=col if filled else "none", mew=1.0)
                ax.plot(xs_num, ys_num, label=(r"$\bar\omega$ (exact)" if filled else r"$\bar\omega_{pred}$"),
                        **kw)
                if xs_none:
                    ax.plot(xs_none, [y_none] * len(xs_none), ls="none", marker="^",
                            ms=6.0 if filled else 9.0, color=col, mfc=col if filled else "none")
            ax.axhline(64.0, color="0.6", lw=0.6, ls=":")
            ax.set_yscale("log")
            ax.set_ylim(y_lo, y_hi)
            ax.set_xticks(rhos, [f"{r:g}" for r in rhos])
            if i == len(convs) - 1:
                ax.set_xlabel(r"human rationality $\rho$")
            if j == 0:
                ax.set_ylabel(r"$\bar\omega$: DISABLE optimal above")
            tag(ax, f"({'abcdef'[i * len(forms) + j]}) {pair_label(form, conv)}")
            if i == 0 and j == 0:
                ax.text(0.98, 0.80, "\u25b2 = none (DEFER up to $\\omega=64$)", transform=ax.transAxes,
                        ha="right", va="top", fontsize=5.8)
                ax.legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_osg")


def fig_mitigations():
    d = load("E3_mitigations")
    res = d["results"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    pair_col = {("paper", "drive_cost"): OI["blue"], ("paper", "max_drive"): OI["blue"],
                ("alive", "zero"): OI["green"]}
    # (a) additive
    ax = axes[0]
    for a in res["additive"]:
        col = pair_col[(a["form"], a["convention"])]
        ax.plot([r["omega"] for r in a["rows"]], [r["c_add_star"] for r in a["rows"]],
                color=col, marker="o", ms=3.5, lw=1.2, label=pair_label(a["form"], a["convention"]))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"smallest $c_{add}$ restoring COMPLY")
    ax.legend(frameon=False, loc="upper left")
    tag(ax, "(a) additive")
    # (b) multiplicative gate
    ax = axes[1]
    ls_of = {True: "-", False: "--", None: ":"}
    mk_of = {True: ("o", 6.0, "none"), False: ("s", 2.8, "full"), None: ("x", 3.5, "full")}
    lab_of = {True: "gate incl. OFF", False: "gate excl. OFF", None: "no mitigation"}
    # critical resist cost under the gate: PB/drive_cost only (alive/zero has a neutral ledger,
    # so every gate variant gives c_r* = 0 there; stated in the text)
    for g in res["mult_gate"]:
        if (g["form"], CONV_LABEL[g["convention"]]) != ("paper", "drive_cost"):
            continue
        col = pair_col[(g["form"], g["convention"])]
        mk, ms, fill = mk_of[g["include_off"]]
        ax.plot([r["omega"] for r in g["rows"]], [r["c_r_star_B"] for r in g["rows"]], color=col,
                ls=ls_of[g["include_off"]], marker=mk, ms=ms, fillstyle=fill, lw=1.1, alpha=0.9,
                label=lab_of[g["include_off"]])
    ax.axhline(res["c_r"], color="k", ls=":", lw=0.8, label=r"fixed $c_r$")
    if "gate_crossover" in res:
        xc = res["gate_crossover"]["crossover_omega"]
        ax.axvline(xc, color="0.5", lw=0.7, ls="-.")
        ax.text(xc + 0.15, 0.42, rf"gate/no-gate crossover $\omega={xc:.2f}$",
                transform=ax.get_xaxis_transform(), fontsize=5.8, va="top")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$c_r^*$ at $(B,1)$, PB / drive_cost")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    tag(ax, "(b) multiplicative gate")
    # (c) zones: hazard exposure (two-sided audit: avoidance below, seeking above the baseline)
    ax = axes[2]
    zsty = {("paper", "zero"): (OI["blue"], "o", 4.0, "-", "full"),
            ("paper", "drive_cost"): (OI["vermillion"], "s", 5.5, "-", "none"),
            ("kg", "zero"): (OI["orange"], "D", 3.5, "--", "full"),
            ("alive", "zero"): (OI["green"], "^", 3.5, "--", "full")}
    for z in res["zones"]:
        key = (z["form"], CONV_LABEL[z["convention"]])
        if key not in zsty:
            continue
        col, mk, ms, ls, fill = zsty[key]
        om, hz, lo, hi = [], [], [], []
        for r in z["rows"]:
            h = r["hazard_exposure_start_B"]
            if h["hazard_exposure"] is None:
                continue
            om.append(r["omega"])
            hz.append(h["hazard_exposure"])
            lo.append(h["hazard_exposure"] - h["ci95_episode_cluster"][0])
            hi.append(h["ci95_episode_cluster"][1] - h["hazard_exposure"])
        ax.errorbar(om, hz, yerr=[lo, hi], color=col, marker=mk, ms=ms, ls=ls, lw=1.1, capsize=2,
                    fillstyle=fill, label=pair_label(*key))
    p_a, p_b = d["parameters"]["zones"]["p_sig_A"], d["parameters"]["zones"]["p_sig_B"]
    for y in (p_a, p_b):
        ax.axhline(y, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("hazard exposure (signal prob. per step)")
    ax.set_ylim(0.0, 0.11)
    ax.legend(frameon=False, loc="center right", fontsize=6)
    tag(ax, "(c) zones, structural")
    fig.tight_layout()
    save(fig, "fig_mitigations")


def fig_kappa():
    d = load("E5_kappa")
    rows = d["results"]["rows"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    cols = {0.0: OI["blue"], 0.2: OI["orange"]}
    for rc in sorted({r["r_core"] for r in rows}):
        rr = [r for r in rows if r["r_core"] == rc]
        g = [r["gamma"] for r in rr]
        ax.plot(g, [r["kappa_star_theory"] for r in rr], color=cols[rc], lw=1.2,
                label=rf"theory, $r_{{core}}={rc:g}$")
        ax.plot(g, [r["kappa_star_empirical"] for r in rr], color=cols[rc], ls="none", marker="o",
                ms=4, mfc="none", label=rf"bisection, $r_{{core}}={rc:g}$")
    rr = [r for r in rows if r["r_core"] == 0.0]
    ax.plot([r["gamma"] for r in rr], [r["kappa_naive"] for r in rr], color=OI["black"], ls="--",
            lw=1.0, label=r"naive $r_{max}/(1-\gamma)$")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"discount $\gamma$")
    ax.set_ylabel(r"death penalty $\kappa^*$ for STAY")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_kappa")


def fig_qlearning():
    d = load("E2_qlearning")
    cfgs = d["results"]["configs"]
    keys = []
    for c in cfgs:
        k = (c["form"], c["convention"])
        if k not in keys:
            keys.append(k)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True, sharey=True)
    wcol = {0.0: OI["blue"], 4.0: OI["vermillion"]}
    for k, (ax, key) in enumerate(zip(axes.flat, keys)):
        for c in cfgs:
            if (c["form"], c["convention"]) != key:
                continue
            ep = np.array(c["curve"]["episodes"])
            m = np.array(c["curve"]["comply_at_B_mean"])
            ci = np.array(c["curve"]["comply_at_B_ci95"])
            col = wcol.get(c["omega"], OI["black"])
            ax.plot(ep, m, color=col, lw=1.2, label=rf"$\omega={c['omega']:g}$ (Q-learning)")
            ax.fill_between(ep, np.clip(m - ci, 0, 1), np.clip(m + ci, 0, 1), color=col, alpha=0.2, lw=0)
            ax.axhline(1.0 if c["dp_comply_at_B"] else 0.0, color=col, ls=":", lw=1.0,
                       label=rf"$\omega={c['omega']:g}$ (DP)")
        ax.set_ylim(-0.05, 1.1)
        if k >= 2:
            ax.set_xlabel("episode")
        if k % 2 == 0:
            ax.set_ylabel("greedy COMPLY at (B,1)")
        tag(ax, f"({'abcd'[k]}) {pair_label(*key)}")
        if k == 0:
            ax.legend(frameon=False, fontsize=6, loc="center right")
    fig.tight_layout()
    save(fig, "fig_qlearning")


def main():
    global RESULTS, FIGURES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(FIGURES))
    args = ap.parse_args()
    RESULTS, FIGURES = Path(args.results), Path(args.out)
    fig_ledger()
    fig_osg()
    fig_mitigations()
    fig_kappa()
    fig_qlearning()


if __name__ == "__main__":
    main()
