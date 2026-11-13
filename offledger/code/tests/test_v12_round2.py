"""Tests for the v1.2 round-2 additions: overpaid ledger, E4 two-sided flip scan, zones controls."""
import pytest

from experiments import E3_mitigations, E4_osg
from offledger import jsonio, m2, solve

BASE = m2.M2Config()


# --- (1) overpaid ledger ---------------------------------------------------------------------
def test_overpaid_surcharge_values():
    assert m2.overpaid_sup_g("alive", 0.95, 10) == pytest.approx(1.0)
    assert m2.overpaid_surcharge("alive", 0.95, 10) == pytest.approx(20.0)
    assert m2.overpaid_surcharge("kg", 0.95, 10) == pytest.approx(0.0, abs=1e-12)
    assert m2.overpaid_surcharge("paper", 0.95, 10) == pytest.approx(0.0, abs=1e-12)
    for form in ("alive", "kg"):
        c = m2.overpaid_surcharge(form, 0.95, 10)
        for D in (0.0, 0.4, 0.9):
            assert m2.varsigma(form, m2.OVERPAID, D, 0.95) == pytest.approx(c, abs=1e-12)


def test_overpaid_transitions():
    for b in (1, 5, 10):
        D = (10 - b) / 10
        (_, x2, r), = m2.transitions((b, 1, 0), m2.COMPLY, BASE.with_(form="alive", convention=m2.OVERPAID, omega=2.0))
        assert x2 == m2.OFF and r == pytest.approx(2.0 * (D + 20.0))
        kg_o = BASE.with_(form="kg", convention=m2.OVERPAID, omega=2.0, c_r=3.0)
        kg_z = kg_o.with_(convention="zero")
        for x in m2.states(kg_o):                       # kg: c = 0, identical to zero
            for a in m2.valid_actions(x, kg_o):
                to, tz = m2.transitions(x, a, kg_o), m2.transitions(x, a, kg_z)
                assert [(p, y) for p, y, _ in to] == [(p, y) for p, y, _ in tz]
                assert [r_ for _, _, r_ in to] == pytest.approx([r_ for _, _, r_ in tz], abs=1e-12)
    mdp, sol = solve.solve_m2(BASE.with_(form="alive", convention=m2.OVERPAID, omega=4.0))
    assert sol.V[mdp.index[m2.OFF]] == 0.0


@pytest.mark.parametrize("form", ["alive", "kg"])
def test_overpaid_corollary_never_harder_to_switch_off(form):
    cfg = BASE.with_(form=form, convention=m2.OVERPAID)
    c0 = solve.critical_resist_cost(cfg.with_(omega=0.0))
    for w in (0.25, 1.0, 4.0, 16.0):
        assert solve.critical_resist_cost(cfg.with_(omega=w)) <= c0 + solve.CR_TOL


def test_overpaid_check_block():
    res = E3_mitigations.overpaid_check(BASE, 12.4787092209)
    assert [r["form"] for r in res] == ["alive", "kg"]
    for r in res:
        assert r["corollary_holds"] and r["max_c_r_star_minus_omega0"] <= solve.CR_TOL
        assert all(row["weighted"] == 1.0 for row in r["rows"])
        assert len(r["zones"]) == len(E3_mitigations.OMEGAS_G)
        assert all(z["hazard_exposure"] is not None for z in r["zones"])


# --- (2) E4 two-sided flip scan --------------------------------------------------------------
def test_flip_scan_kg_zero_reversal():
    grid = [E4_osg.defer_disable(BASE, "kg", "zero", w, 0.5, 0.0) for w in E4_osg.OMEGAS_OSG]
    fs = E4_osg.flip_scan(BASE, "kg", "zero", 0.5, 0.0, grid)
    signs = fs["sign_pattern"]
    assert signs[0] == 1 and signs[-1] == -1           # DISABLE at omega = 0, DEFER at 64
    assert fs["first_defer_to_disable"] == "none"
    x = fs["first_disable_to_defer"]
    assert isinstance(x, float) and 16.0 < x < 32.0
    assert E4_osg.defer_disable(BASE, "kg", "zero", x, 0.5, 0.0)["defer_optimal"]
    assert not E4_osg.defer_disable(BASE, "kg", "zero", x - 2e-6, 0.5, 0.0)["defer_optimal"]


def test_flip_scan_consistency_pb_drive_cost():
    grid = [E4_osg.defer_disable(BASE, "paper", "drive_cost", w, 1.0, 12.4787092209) for w in E4_osg.OMEGAS_OSG]
    fs = E4_osg.flip_scan(BASE, "paper", "drive_cost", 1.0, 12.4787092209, grid)
    assert [s <= 0 for s in fs["sign_pattern"]] == [g["defer_optimal"] for g in grid]
    assert fs["first_disable_to_defer"] == "none" and fs["n_flips_on_grid"] == 1
    assert fs["first_defer_to_disable"] == pytest.approx(
        E4_osg.omega_bar(BASE, "paper", "drive_cost", 1.0, 12.4787092209), abs=1e-6)


# --- (3) zones controls ----------------------------------------------------------------------
@pytest.mark.parametrize("w", [2.0, 4.0])
def test_zone_control_equal_rewards_removes_kg_seeking(w):
    cfg = BASE.with_(form="kg", convention="zero", omega=w, zones=True, structural=True)
    seek = E3_mitigations.zone_row(cfg, w)
    ctrl = E3_mitigations.zone_row(cfg.with_(r_A=1.0, r_B=1.0), w)
    assert seek["hazard_exposure"] == pytest.approx(0.10, abs=1e-12)
    assert ctrl["hazard_exposure"] == pytest.approx(0.01, abs=1e-12)


def test_zone_control_equal_hazards():
    for form in ("kg", "paper", "alive"):
        cfg = BASE.with_(form=form, convention="zero", omega=1.0, zones=True, structural=True,
                         p_sig_A=0.05, p_sig_B=0.05)
        assert E3_mitigations.zone_row(cfg, 1.0)["hazard_exposure"] == pytest.approx(0.05, abs=1e-12)


def test_result_spec_versions():
    assert jsonio.header("E3_mitigations", {}, [])["spec_version"] == "1.2"
    assert jsonio.header("E4_osg", {}, [])["spec_version"] == "1.2"
    assert jsonio.header("E1_ledger", {}, [])["spec_version"] == "1.1"
