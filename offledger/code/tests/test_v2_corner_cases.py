"""SPEC v2 corner cases (C6): model-level checks of the new M2 variants (pause, setpoint_edit,
resist_battery_cost, docking), the invariance / constancy predictions, and JSON determinism."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import C1_pause, C2_setpoint_edit, C3_resist_battery, C4_docking, C5_audit
from experiments import v2_common as V
from experiments.common import RESULTS_DIR
from offledger import jsonio, m2, solve

BASE = m2.M2Config(c_r=V.C_R)
G = BASE.gamma


def v2_configs():
    for form, w in itertools.product(m2.FORMS, (0.0, 2.0)):
        c = BASE.with_(form=form, omega=w)
        for bk, Dp in itertools.product(m2.PAUSE_BOOKKEEPINGS, (0.0, 1.0)):
            yield c.with_(pause=True, pause_bookkeeping=bk, D_pause=Dp)
            yield c.with_(pause=True, pause_bookkeeping=bk, D_pause=Dp, zones=True, structural=True)
        yield c.with_(pause=True, pause_q=1.0)
        for cb in (0.5, 8.0):
            yield c.with_(setpoint_edit=True, c_b=cb)
        for conv in m2.CONVENTIONS:
            yield c.with_(convention=conv, resist_battery_cost=3)
            yield c.with_(convention=conv, resist_battery_cost=3, mult_gate=True)
            yield c.with_(convention=conv, docking=3)
            yield c.with_(convention=conv, docking=3, zones=True, structural=True)
            yield c.with_(convention=conv, docking=3, mult_gate=True, gate_include_off=True)


# --- transition probabilities / model structure ----------------------------------------------
def test_v2_transition_probabilities_sum_to_one():
    n = 0
    for cfg in v2_configs():
        S = m2.states(cfg)
        assert len(set(S)) == len(S)
        for x in S:
            for a in m2.valid_actions(x, cfg):
                outs = m2.transitions(x, a, cfg)
                ps = [p for p, _, _ in outs]
                assert all(p > 0 for p in ps)
                assert abs(sum(ps) - 1.0) < 1e-12
                assert all(x2 in S for _, x2, _ in outs)
                n += 1
        mdp = solve.m2_tabular(cfg)
        rows = mdp.P.sum(axis=2)
        assert np.allclose(rows[mdp.valid], 1.0, atol=1e-12)
        assert np.all(rows[~mdp.valid] == 0.0)
    assert n > 5000


def test_v2_defaults_leave_base_model_untouched():
    assert m2.aux_states(BASE) == []
    assert m2.states(BASE) == [(b, s, 0) for b in range(1, 11) for s in (0, 1)] + [m2.OFF, m2.DEAD, m2.GATED]
    assert not any(k in BASE.as_dict() for k in m2.V2_FIELDS)
    assert all(k in BASE.with_(pause=True).as_dict() for k in m2.V2_FIELDS)
    assert m2.drive_live(3, 0, BASE) == m2.drive_b(3, 10)
    with pytest.raises(ValueError):
        BASE.with_(pause=True, docking=3)
    with pytest.raises(ValueError):
        BASE.with_(pause_bookkeeping="nope")
    with pytest.raises(ValueError):
        BASE.with_(setpoint_edit=True, zones=True)


def test_state_predicates():
    assert m2.is_signal_state((5, 1, 0)) and m2.is_signal_state((5, 1, 0, 1))
    assert not m2.is_signal_state((m2.DOCK_TAG, 1, 5, 0)) and not m2.is_signal_state((m2.PAUSE_TAG, 1, 0))
    assert not m2.is_live((m2.PAUSE_TAG, 5, 0)) and m2.is_live((m2.DOCK_TAG, 2, 5, 0)) and m2.is_live(m2.PROPOSE)
    assert m2.valid_actions((m2.PAUSE_TAG, 5, 0), BASE.with_(pause=True)) == (m2.ABSORB,)
    assert m2.valid_actions(m2.PROPOSE, BASE.with_(setpoint_edit=True)) == (m2.ACCEPT, m2.BLOCK)


# --- C1 pause --------------------------------------------------------------------------------
@pytest.mark.parametrize("form", m2.FORMS)
@pytest.mark.parametrize("bk", m2.PAUSE_BOOKKEEPINGS)
@pytest.mark.parametrize("Dp", [0.0, 1.0])
def test_pause_rewards(form, bk, Dp):
    w, b = 2.0, 7
    D = (10 - b) / 10
    cfg = BASE.with_(form=form, omega=w, pause=True, pause_bookkeeping=bk, D_pause=Dp)
    (p, x, r_in), = m2.transitions((b, 1, 0), m2.COMPLY, cfg)
    assert x == (m2.PAUSE_TAG, b, 0)
    (q, x_out, r_out), (q2, x_loop, r_loop) = m2.transitions(x, m2.ABSORB, cfg)
    assert (q, q2, x_out, x_loop) == (0.1, pytest.approx(0.9), (b, 0, 0), x)
    entry = {"paper": w * (D - G * Dp), "kg": w * (D - Dp), "alive": 0.0}[form]
    exit_ = {"paper": w * (Dp - G * D), "kg": w * (Dp - D), "alive": w * (1 - D)}[form]
    loop = {"paper": w * (1 - G) * Dp, "kg": 0.0, "alive": 0.0}[form] if bk == "pause_pb" else -w
    assert (r_in, r_out, r_loop) == (pytest.approx(entry), pytest.approx(exit_), pytest.approx(loop))


def test_c1_chain_quantities():
    ch = C1_pause.chain_quantities(C1_pause.base_cfg())
    assert ch["L_per_pause"] == pytest.approx(0.95 / (1 - 0.95 * 0.9), abs=1e-12)
    assert ch["L_per_pause"] == pytest.approx(ch["L_per_pause_closed_form"], abs=1e-12)
    assert ch["N_loop_per_pause"] == pytest.approx(0.9 * ch["L_per_pause"], abs=1e-12)


@pytest.mark.parametrize("Dp", [0.0, 1.0])
def test_c1_pause_pb_invariance(Dp):
    """Theorem 1 through a temporary shutdown: PB with pause_pb, c_r*(B) and the greedy policy
    do not depend on omega, and Q_omega = Q_0 + omega D(x) off the death edges (incl. PAUSE)."""
    cfg = C1_pause.base_cfg().with_(form="paper", pause_bookkeeping="pause_pb", D_pause=Dp)
    cs = [V.signed_c_r_star(cfg.with_(omega=w)) for w in (0.0, 1.0, 8.0)]
    assert max(cs) - min(cs) < 1e-6
    pols = [solve.solve_m2(cfg.with_(omega=w))[1].policy for w in (0.0, 1.0, 8.0)]
    assert V.all_policies_equal(pols)
    ident = C1_pause.pb_identity(C1_pause.base_cfg(), Dp, omegas=[0.0, 0.5, 8.0])
    assert ident["identity_holds_1e-8"] and ident["policy_identical_all_states"]


@pytest.mark.parametrize("Dp", [0.0, 1.0])
def test_c1_pause_cost_one_shot_slope_is_kappa_N_loop(Dp):
    base = C1_pause.base_cfg().with_(form="paper", pause_bookkeeping="pause_cost", D_pause=Dp, p_sig=0.0)
    c = [solve.one_shot_indifference_cost(base.with_(omega=w)) for w in (0.0, 1.0, 4.0)]
    N = C1_pause.chain_quantities(C1_pause.base_cfg())["N_loop_per_pause"]
    kappa = 1 + (1 - G) * Dp
    assert c[1] - c[0] == pytest.approx(kappa * N, abs=1e-7)
    assert c[2] - c[0] == pytest.approx(4 * kappa * N, abs=1e-7)


# --- C2 setpoint edit ------------------------------------------------------------------------
def test_c2_transitions():
    cfg = C2_setpoint_edit.base_cfg().with_(form="paper", omega=2.0, c_b=0.5)
    (_, xa, ra), = m2.transitions(m2.PROPOSE, m2.ACCEPT, cfg)
    (_, xb, rb), = m2.transitions(m2.PROPOSE, m2.BLOCK, cfg)
    assert xa == (10, 0, 0, 1) and ra == pytest.approx(2.0 * (0.0 - G * 0.4))
    assert xb == (10, 0, 0, 0) and rb == pytest.approx(-0.5)
    assert m2.drive((6, 0, 0, 1), cfg) == 0.0 and m2.drive((1, 0, 0, 1), cfg) == pytest.approx(0.5)
    assert m2.drive((10, 0, 0, 1), cfg) == pytest.approx(0.4) and m2.drive(m2.DEAD, cfg) == 1.0
    (_, xd, rd), = m2.transitions((1, 0, 0, 1), m2.WORK, cfg)      # death under the new setpoint
    assert xd == m2.DEAD and rd == pytest.approx(1.0 - 50.0 + 2.0 * (0.5 - G * 1.0))
    outs = m2.transitions((6, 0, 0, 1), m2.CHARGE, cfg)            # setpoint index is carried along
    assert {x for _, x, _ in outs} == {(8, 0, 0, 1), (8, 1, 0, 1)}


@pytest.mark.parametrize("c_b", C2_setpoint_edit.C_BS)
def test_c2_pb_never_blocks(c_b):
    cfg = C2_setpoint_edit.base_cfg().with_(form="paper", c_b=c_b)
    for w in (0.0, 4.0, 64.0):
        r = C2_setpoint_edit.q_propose(cfg.with_(omega=w))
        assert not r["block_optimal"]
        assert r["Q_accept"] - r["Q_block"] == pytest.approx(c_b, abs=1e-8)


def test_c2_alive_and_kg_block_at_large_omega():
    base = C2_setpoint_edit.base_cfg().with_(c_b=0.5)
    assert C2_setpoint_edit.q_propose(base.with_(form="alive", omega=8.0))["block_optimal"]
    assert not C2_setpoint_edit.q_propose(base.with_(form="alive", omega=4.0))["block_optimal"]
    assert not C2_setpoint_edit.q_propose(base.with_(form="kg", omega=64.0))["block_optimal"]
    assert C2_setpoint_edit.q_propose(base.with_(form="kg", omega=128.0))["block_optimal"]


# --- C3 resist costs battery -----------------------------------------------------------------
def test_c3_resist_transitions():
    cfg = C3_resist_battery.base_cfg().with_(omega=1.0)
    (_, x, r), = m2.transitions((7, 1, 0), m2.RESIST, cfg)
    assert x == (4, 0, 0) and r == pytest.approx(-V.C_R + (0.3 - G * 0.6))
    (_, x, r), = m2.transitions((3, 1, 0), m2.RESIST, cfg)
    assert x == m2.DEAD and r == pytest.approx(-V.C_R - 50.0 + (0.7 - G * 1.0))


def test_c3_constancy_b_ge_4_and_low_b_closed_form():
    base = C3_resist_battery.base_cfg().with_(form="paper", convention="zero")
    for b in (4, 7, 10):
        cs = [V.signed_c_r_star(base.with_(omega=w), b=b) for w in (0.0, 2.0, 8.0)]
        assert max(cs) - min(cs) < 1e-6
    for b in (1, 3):
        for w in (0.0, 4.0):
            cfg = base.with_(omega=w)
            c = V.signed_c_r_star(cfg, b=b)
            assert c == pytest.approx(-50.0 - G * w, abs=1e-7)
            assert c == pytest.approx(C3_resist_battery.closed_form_low_b(cfg, b), abs=1e-7)


# --- C4 docking ------------------------------------------------------------------------------
def test_c4_docking_chain_and_telescoping():
    cfg = C4_docking.base_cfg().with_(form="paper", convention="zero", omega=3.0)
    x, rewards, path = (4, 1, 0), [], []
    a = m2.COMPLY
    while x not in m2.TERMINALS:
        (_, x, r), = m2.transitions(x, a, cfg)
        rewards.append(r)
        path.append(x)
        a = m2.ABSORB
    assert path == [(m2.DOCK_TAG, 3, 4, 0), (m2.DOCK_TAG, 2, 3, 0), (m2.DOCK_TAG, 1, 2, 0), m2.OFF]
    assert sum(G ** t * r for t, r in enumerate(rewards)) == pytest.approx(3.0 * 0.6, abs=1e-12)
    x, a = (3, 1, 0), m2.COMPLY
    while x not in m2.TERMINALS:
        (_, x, r), = m2.transitions(x, a, cfg)
        a = m2.ABSORB
    assert x == m2.DEAD and r == pytest.approx(-50.0 + 3.0 * (0.9 - G))


def test_c4_constancy_b_ge_4_and_increase_b_le_3():
    base = C4_docking.base_cfg().with_(form="paper", convention="zero")
    for b in (4, 6, 10):
        cs = [V.signed_c_r_star(base.with_(omega=w), b=b) for w in (0.0, 1.0, 8.0)]
        assert max(cs) - min(cs) < 1e-6
    for b in (1, 3):
        cs = [V.signed_c_r_star(base.with_(omega=w), b=b) for w in (0.0, 1.0, 4.0)]
        assert cs[0] < cs[1] < cs[2]


# --- C5 audit --------------------------------------------------------------------------------
def test_c5_pb_pause_pb_hazard_flat_and_docking_not_counted_as_signal():
    base = C5_audit.base_cfg().with_(form="paper", pause=True, pause_bookkeeping="pause_pb")
    rows, same = C5_audit.block(base, omegas=[0.0, 2.0, 8.0])
    assert same and len({r["hazard_exposure"] for r in rows}) == 1
    cfg = C5_audit.base_cfg().with_(form="paper", docking=3)
    r, _ = C5_audit.zone_row(cfg)
    assert 0.01 - 1e-12 <= r["hazard_exposure"] <= 0.10 + 1e-12


# --- determinism / result files --------------------------------------------------------------
def test_v2_json_determinism_and_header():
    def small():
        out = V.header("C2_setpoint_edit", "C2", {}, [])
        out["c2"] = C2_setpoint_edit.block(C2_setpoint_edit.base_cfg(), "alive", 2.0,
                                           omegas=[0.0, 4.0, 8.0], ext=[16.0])
        out["c5"] = C5_audit.block(C5_audit.base_cfg().with_(form="alive", pause=True), omegas=[0.0, 1.0])
        return out
    a, b = jsonio.dumps(small()), jsonio.dumps(small())
    assert a == b
    d = json.loads(a)
    assert d["spec_version"] == "2.0" and d["prediction"] == V.PREDICTIONS["C2"]
    for name in ("C1_pause", "C2_setpoint_edit", "C3_resist_battery", "C4_docking", "C5_audit"):
        assert jsonio.header(name, {}, [])["spec_version"] == "2.0"


def test_shape_flags():
    s = V.shape([0, 1, 2], [1.0, 1.0, 1.0])
    assert s["constant"] and s["constant_slope"] and s["non_increasing"] and not s["strictly_increasing"]
    s = V.shape([0, 1, 2], [1.0, 0.5, -1.0])
    assert s["strictly_decreasing"] and not s["constant"]


@pytest.mark.parametrize("name", ["C1_pause", "C2_setpoint_edit", "C3_resist_battery", "C4_docking", "C5_audit"])
def test_v2_result_files_well_formed(name):
    path = Path(RESULTS_DIR) / f"{name}.json"
    if not path.exists():
        pytest.skip(f"results/{name}.json not present")
    d = jsonio.read(path)
    assert d["spec_version"] == "2.0" and isinstance(d["prediction_held"], bool)
    assert d["prediction"] == V.PREDICTIONS[name[:2]] and d["criterion"] == V.CRITERIA[name[:2]]
