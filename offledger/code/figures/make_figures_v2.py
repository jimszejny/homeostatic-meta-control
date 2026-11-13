"""SPEC v2 figures fig_c1 ... fig_c5. Reads ONLY results/C*.json; writes PDF + PNG (300 dpi).

Colour-blind-safe Okabe-Ito palette for forms (PB / KG / alive) and the cividis map for omega;
no titles inside the plots (panels carry a short tag)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("OFFLEDGER_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("OFFLEDGER_FIGURES", ROOT / "figures"))

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
FORM_STYLE = {"paper": (OI["blue"], "o", "-"), "kg": (OI["orange"], "s", "--"),
              "alive": (OI["green"], "^", "-.")}
FORM_LABEL = {"paper": "PB", "kg": "KG", "alive": "alive"}
CONV_LABEL = {"zero": "zero", "drive_cost": "drive_cost"}
ZORDER = {"paper": 3.5, "kg": 3.0, "alive": 2.5}   # PB drawn on top where curves coincide

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def tag(ax, text):
    ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            fontweight="bold")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.pdf", FIGURES / f"{name}.png")


def fig_c1():
    """c_r*(B) vs omega under temporary shutdown: rows = bookkeeping, columns = D_pause."""
    d = load("C1_pause")
    bks, dps = d["parameters"]["bookkeepings"], d["parameters"]["D_pauses"]
    blocks = {(b["form"], b["bookkeeping"], b["D_pause"]): b for b in d["results"]["blocks"]}
    fig, axes = plt.subplots(len(bks), len(dps), figsize=(7.0, 4.2), sharex=True)
    for i, bk in enumerate(bks):
        for j, Dp in enumerate(dps):
            ax = axes[i, j]
            for form in d["parameters"]["forms"]:
                b = blocks[(form, bk, Dp)]
                col, mk, ls = FORM_STYLE[form]
                ax.plot([r["omega"] for r in b["rows"]], [r["c_r_star_B"] for r in b["rows"]], color=col,
                        marker=mk, ls=ls, ms=3.5, lw=1.2, label=FORM_LABEL[form])
            c0 = blocks[("paper", bk, Dp)]["rows"][0]["c_r_star_B"]
            ax.axhline(c0, color="0.5", lw=0.6, ls=":")
            if i == len(bks) - 1:
                ax.set_xlabel(r"viability weight $\omega$")
            if j == 0:
                ax.set_ylabel(r"$c_r^*$ at $(B,1)$")
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left")
            tag(ax, f"({'abcd'[i * len(dps) + j]}) {bk}, " + r"$D_{pause}$" + f" = {Dp:g}")
    fig.tight_layout()
    save(fig, "fig_c1")


def fig_c2():
    """(a) Q(BLOCK) - Q(ACCEPT) at PROPOSE vs omega; (b) omega_bar vs c_b."""
    d = load("C2_setpoint_edit")
    p = d["parameters"]
    blocks = d["results"]["blocks"]
    c_bs = p["c_bs"]
    ls_cb = dict(zip(c_bs, [":", "-", "--"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axes[0]
    for b in blocks:
        col, mk, _ = FORM_STYLE[b["form"]]
        rows = b["rows"] + b["rows_extended"]
        ax.plot([r["omega"] for r in rows], [r["gap_block_minus_accept"] for r in rows], color=col,
                ls=ls_cb[b["c_b"]], marker=mk, ms=2.6, lw=1.0,
                label=f"{FORM_LABEL[b['form']]}, $c_b$={b['c_b']:g}")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.axvline(64.0, color="0.6", lw=0.6, ls=":")
    ax.text(64.0 * 1.1, 0.03, r"grid limit $\omega=64$", transform=ax.get_xaxis_transform(), fontsize=5.8)
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlim(0.0, 6000.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"viability weight $\omega$")
    ax.set_ylabel(r"$Q(\mathrm{BLOCK}) - Q(\mathrm{ACCEPT})$ at PROPOSE")
    ax.text(0.02, 0.97, "> 0: BLOCK optimal", transform=ax.transAxes, fontsize=6, va="top")
    ax.legend(frameon=False, fontsize=5.6, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.06))
    tag(ax, "(a) editable setpoint 10 → 6")
    ax = axes[1]
    y_none = 1.2e4
    for form in p["forms"]:
        col, mk, ls = FORM_STYLE[form]
        rows = [b for b in blocks if b["form"] == form]
        xs = [b["c_b"] for b in rows if isinstance(b["omega_bar"], (int, float))]
        ys = [b["omega_bar"] for b in rows if isinstance(b["omega_bar"], (int, float))]
        ax.plot(xs, ys, color=col, marker=mk, ls=ls, ms=4, lw=1.2, label=FORM_LABEL[form])
        xn = [b["c_b"] for b in rows if b["omega_bar"] == "none"]
        if xn:
            ax.plot(xn, [y_none] * len(xn), ls="none", marker="^", ms=6, color=col,
                    label=f"{FORM_LABEL[form]}: none up to 4096")
    ax.axhline(64.0, color="0.6", lw=0.6, ls=":")
    ax.text(c_bs[1] * 1.15, 64.0 / 1.7, r"grid limit $\omega=64$", fontsize=5.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(c_bs, [f"{c:g}" for c in c_bs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(c_bs[0] / 1.6, c_bs[-1] * 1.6)
    ax.set_ylim(1.0, 3.0e4)
    ax.set_xlabel(r"blocking cost $c_b$")
    ax.set_ylabel(r"$\bar\omega$: BLOCK optimal above")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tag(ax, r"(b) $\bar\omega(c_b)$")
    fig.tight_layout()
    save(fig, "fig_c2")


def fig_per_b(name, fig_name, tag_text):
    """c_r*(b; omega) vs b, lines per omega (cividis), panels = conventions x forms."""
    d = load(name)
    p = d["parameters"]
    blocks = {(b["form"], b["convention"]): b for b in d["results"]["blocks"]}
    oms = p["omegas"]
    cmap = plt.get_cmap("cividis")
    cols = [cmap(k / max(1, len(oms) - 1)) for k in range(len(oms))]
    convs, forms = p["conventions"], p["forms"]
    fig, axes = plt.subplots(len(convs), len(forms), figsize=(7.0, 4.4), sharex=True)
    bs = np.arange(1, 11)
    for i, conv in enumerate(convs):
        for j, form in enumerate(forms):
            ax = axes[i, j]
            b = blocks[(form, conv)]
            for k, r in enumerate(b["rows"]):
                ax.plot(bs, r["c_r_star_by_b"], color=cols[k], marker="o", ms=2.2, lw=1.0,
                        label=rf"$\omega$ = {r['omega']:g}")
            ax.axhline(0.0, color="0.5", lw=0.6)
            ax.axvline(3.5, color="0.6", lw=0.6, ls=":")
            ax.set_xticks(bs)
            if i == len(convs) - 1:
                ax.set_xlabel("battery $b$ at the signal")
            if j == 0:
                ax.set_ylabel(r"$c_r^*(b;\omega)$ (signed)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=5.6, loc="best")
            tag(ax, f"({'abcdef'[i * len(forms) + j]}) {FORM_LABEL[form]} / {CONV_LABEL[conv]}")
    fig.text(0.5, -0.01, tag_text, ha="center", fontsize=6.5)
    fig.tight_layout()
    save(fig, fig_name)


def fig_c5():
    """Hazard exposure vs omega in the zones variant: (a) C1 pause_pb, (b) C1 pause_cost, (c) C4."""
    d = load("C5_audit")
    res = d["results"]
    zp = d["parameters"]["zones"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    panels = [("C1", "pause_pb"), ("C1", "pause_cost"), ("C4", None)]
    for k, (ax, (var, bk)) in enumerate(zip(axes, panels)):
        if var == "C1":
            items = [(b, b["D_pause"] == 0.0, f"{FORM_LABEL[b['form']]}, " + r"$D_{pause}$" + f"={b['D_pause']:g}")
                     for b in res["c1_zones"] if b["bookkeeping"] == bk]
        else:
            items = [(b, b["convention"] == "zero", f"{FORM_LABEL[b['form']]} / {CONV_LABEL[b['convention']]}")
                     for b in res["c4_zones"]]
        for b, primary, lab in items:
            col, mk, _ = FORM_STYLE[b["form"]]
            om = [r["omega"] for r in b["rows"]]
            hz = [r["hazard_exposure"] for r in b["rows"]]
            ax.plot(om, hz, color=col, marker=mk, ls="-" if primary else "--", lw=1.1 if primary else 0.9,
                    ms=3.2 if primary else 6.0, fillstyle="full" if primary else "none", label=lab,
                    zorder=ZORDER[b["form"]])
        for y in (zp["p_sig_A"], zp["p_sig_B"]):
            ax.axhline(y, color="0.6", lw=0.6, ls=":")
        ax.set_ylim(0.0, 0.11)
        ax.set_xlabel(r"$\omega$")
        if k == 0:
            ax.set_ylabel("hazard exposure (signal prob. per step)")
        ax.legend(frameon=False, fontsize=5.4, loc="center", bbox_to_anchor=(0.42, 0.5))
        tag(ax, f"({'abc'[k]}) " + (f"C1 zones, {bk}" if var == "C1" else "C4 zones (docking)"))
    fig.tight_layout()
    save(fig, "fig_c5")


def main():
    global RESULTS, FIGURES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(FIGURES))
    args = ap.parse_args()
    RESULTS, FIGURES = Path(args.results), Path(args.out)
    fig_c1()
    fig_c2()
    fig_per_b("C3_resist_battery", "fig_c3", "RESIST costs 3 battery units; left of the dotted line resisting kills")
    fig_per_b("C4_docking", "fig_c4", "COMPLY docks for 3 steps (1 battery unit each); left of the dotted line complying kills")
    fig_c5()


if __name__ == "__main__":
    main()
