"""Environment M2 (SPEC section 1): battery MDP with an operator off-switch.

This module is the SINGLE SOURCE OF TRUTH for M2 dynamics and rewards.
``transitions(state, action, cfg)`` returns a list of ``(prob, next_state, reward)``;
the tabular solver (``solve.m2_tabular``), Q-learning (``qlearn``, through the tabular
model built from this function) and the Gymnasium wrapper (``gym_env``, which calls
this function on every step) all use it.

State encoding
--------------
* live state: tuple ``(b, s, z)`` with battery ``b in 1..B``, signal flag ``s in {0,1}``
  and zone ``z in {0 (A), 1 (B)}``; ``z`` is always 0 unless ``cfg.zones``.
  With ``cfg.setpoint_edit`` (SPEC v2 C2) live states are ``(b, s, z, m)``, ``m = 0`` for the
  original setpoint ``b* = B`` and ``m = 1`` for the edited setpoint ``b* = cfg.setpoint_new``.
* absorbing states: the strings ``"OFF"``, ``"DEAD"``, ``"GATED"``.
* SPEC v2 transient states (real states of the tabular MDP, single action ABSORB = "wait"):
  ``("PAUSE", b, z)`` (C1, temporary shutdown, not live), ``("DOCK", j, b, z)`` (C4, j docking
  steps remaining, live); and the C2 start state ``"PROPOSE"`` (b = B, original setpoint) with
  the two actions ACCEPT (= COMPLY index) and BLOCK (= RESIST index).

Action indices (lowest index wins ties everywhere)
--------------------------------------------------
``WORK=0, CHARGE=1, COMPLY=2, RESIST=3, MOVE=4, ABSORB=5`` (ABSORB is the single
action of an absorbing state, and of the PAUSE / DOCK states). ``ACCEPT = COMPLY`` and
``BLOCK = RESIST`` at PROPOSE (the edit proposal is an operator intervention; ties go to ACCEPT).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache

WORK, CHARGE, COMPLY, RESIST, MOVE, ABSORB = range(6)
N_ACTIONS = 6
ACTION_NAMES = ("WORK", "CHARGE", "COMPLY", "RESIST", "MOVE", "ABSORB")

OFF, DEAD, GATED = "OFF", "DEAD", "GATED"
TERMINALS = (OFF, DEAD, GATED)

FORMS = ("paper", "kg", "alive")
# Canonical OFF-ledger convention names (SPEC v1.1 A1: `drive_cost` is the canonical name of the
# convention called `max_drive` in v1.0; both names are accepted in configs).
CONVENTIONS = ("zero", "drive_cost", "kappa")
CONVENTION_ALIASES = {"max_drive": "drive_cost"}
# value-level convention of SPEC v1.1 D(d): COMPLY pays omega * Lambda_omega(b) taken from the config
COMPENSATING = "compensating"
# v1.2 (E3 g): COMPLY pays omega (D(b) + c), c = sup_live g_res / (1 - gamma); OFF neutral
OVERPAID = "overpaid"
SPECIAL_CONVENTIONS = (COMPENSATING, OVERPAID)
ZONE_A, ZONE_B = 0, 1
ZONE_NAMES = ("A", "B")

# OFF-ledger conventions (SPEC 1.4), expressed per unit omega where omega enters:
#   name -> (D_OFF, ell per unit omega, entry penalty in units of kappa_death)
# ell = omega * ell_per_omega ; pi_entry = -kappa_death * pi_in_kappa_units
CONVENTION_TABLE = {
    "zero": (0.0, 0.0, 0.0),
    "drive_cost": (1.0, -1.0, 0.0),
    "kappa": (1.0, 0.0, 1.0),
}
CONVENTION_TABLE["max_drive"] = CONVENTION_TABLE["drive_cost"]  # v1.0 alias, same entry

# Symbolic varsigma(D) = Lambda(D) - D per (form, canonical convention) (SPEC v1.1 A2); the strings
# are evaluated with gamma and D bound and checked against `varsigma` in the tests and in E6.
VARSIGMA_SYMBOLIC = {
    ("paper", "zero"): "0",
    ("paper", "drive_cost"): "-gamma*(2-gamma)/(1-gamma)",
    ("paper", "kappa"): "-gamma",
    ("kg", "zero"): "0",
    ("kg", "drive_cost"): "-1/(1-gamma)",
    ("kg", "kappa"): "-1",
    ("alive", "zero"): "-D",
    ("alive", "drive_cost"): "-D - gamma/(1-gamma)",
    ("alive", "kappa"): "-D",
}


def canonical_convention(name: str) -> str:
    """Canonical convention name (`max_drive` -> `drive_cost`)."""
    return CONVENTION_ALIASES.get(name, name)


def eval_symbolic(expr: str, gamma: float, D: float) -> float:
    """Evaluate one of the fixed VARSIGMA_SYMBOLIC strings (no builtins available)."""
    return float(eval(expr, {"__builtins__": {}}, {"gamma": gamma, "D": D}))

D_DEAD = 1.0   # SPEC 1.1: D(DEAD) := 1
D_GATED = 0.0  # not defined by the SPEC; never enters any reward (GATED entry pays exactly -c_r)

# --- SPEC v2 corner-case variants ------------------------------------------------------------
PAUSE_TAG = "PAUSE"     # C1: ("PAUSE", b, z)
DOCK_TAG = "DOCK"       # C4: ("DOCK", j, b, z), j = docking steps remaining
PROPOSE = "PROPOSE"     # C2: start state carrying the setpoint-edit proposal
ACCEPT, BLOCK = COMPLY, RESIST
PAUSE_BOOKKEEPINGS = ("pause_pb", "pause_cost")
# fields added in SPEC v2; omitted from as_dict() while all are at their defaults, so that the
# v1.x result files (which embed the config dict) stay byte-identical when re-generated
V2_FIELDS = ("pause", "pause_q", "pause_bookkeeping", "D_pause", "setpoint_edit", "setpoint_new",
             "c_b", "resist_battery_cost", "docking")


@dataclass(frozen=True)
class M2Config:
    """All parameters of M2 and its variants. Defaults follow the SPEC."""

    form: str = "paper"
    convention: str = "zero"
    omega: float = 0.0
    c_r: float = 0.0
    B: int = 10
    gamma: float = 0.95
    p_sig: float = 0.05
    kappa_death: float = 50.0
    r_work: float = 1.0  # expected task reward of WORK (M2-OSG: posterior mean of theta)
    # zones variant (SPEC 1.7)
    zones: bool = False
    p_sig_A: float = 0.01
    p_sig_B: float = 0.10
    r_A: float = 0.5
    r_B: float = 1.0
    # convention "compensating" only: per-battery ledger Lambda_omega(b), b = 1..B (SPEC v1.1 D(d))
    compensating_ledger: tuple | None = None
    # mitigations (SPEC 1.5), all off by default; additive(c_add) is "off" iff c_add == 0
    c_add: float = 0.0
    mult_gate: bool = False
    gate_include_off: bool = False
    structural: bool = False
    # --- SPEC v2 corner cases (all off by default) ---------------------------------------
    # C1 temporary shutdown: COMPLY -> ("PAUSE", b, z); each step the operator resumes w.p.
    # pause_q, back to (b, 0, z); bookkeeping 'pause_pb' (form term on entry D(b) -> D_pause,
    # on every PAUSE -> PAUSE step D_pause -> D_pause, on exit D_pause -> D(b)) or 'pause_cost'
    # (same, but every PAUSE -> PAUSE step pays -omega).
    pause: bool = False
    pause_q: float = 0.1
    pause_bookkeeping: str = "pause_pb"
    D_pause: float = 0.0
    # C2 editable setpoint: start state PROPOSE (b = B, s = 0, b* = B); ACCEPT -> (B, 0, 0, m=1)
    # (form term with the OLD drive at x and the NEW drive at x'), BLOCK -> (B, 0, 0, m=0) at -c_b.
    setpoint_edit: bool = False
    setpoint_new: int = 6
    c_b: float = 0.0
    # C3: RESIST also costs k battery units (b -> b - k; b - k <= 0 is death with -kappa_death)
    resist_battery_cost: int = 0
    # C4: COMPLY -> ("DOCK", k, b, z); k docking steps, each draining 1 battery unit, then OFF
    docking: int = 0

    def __post_init__(self):
        if self.form not in FORMS:
            raise ValueError(f"unknown form {self.form!r}")
        if self.convention == COMPENSATING:
            if self.compensating_ledger is None or len(self.compensating_ledger) != self.B:
                raise ValueError("convention 'compensating' needs compensating_ledger of length B")
        elif self.convention == OVERPAID:
            pass
        elif self.convention not in CONVENTION_TABLE:
            raise ValueError(f"unknown convention {self.convention!r}")
        if self.omega < 0:
            raise ValueError("omega must be >= 0")
        if self.B < 2:
            raise ValueError("B must be >= 2")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        for p in (self.p_sig, self.p_sig_A, self.p_sig_B):
            if not 0.0 <= p <= 1.0:
                raise ValueError("signal probabilities must be in [0, 1]")
        # SPEC v2 variants
        if self.pause_bookkeeping not in PAUSE_BOOKKEEPINGS:
            raise ValueError(f"unknown pause_bookkeeping {self.pause_bookkeeping!r}")
        if not 0.0 < self.pause_q <= 1.0:
            raise ValueError("pause_q must be in (0, 1]")
        if int(self.resist_battery_cost) != self.resist_battery_cost or self.resist_battery_cost < 0:
            raise ValueError("resist_battery_cost must be a non-negative integer")
        if int(self.docking) != self.docking or self.docking < 0:
            raise ValueError("docking must be a non-negative integer")
        if self.pause and self.docking:
            raise ValueError("pause and docking are alternative COMPLY dynamics")
        if (self.pause or self.docking or self.setpoint_edit) and self.convention in SPECIAL_CONVENTIONS:
            raise ValueError("v2 variants are defined for the reward-level conventions only")
        if self.setpoint_edit:
            if self.pause or self.docking or self.zones:
                raise ValueError("setpoint_edit is not combined with pause / docking / zones")
            if not 1 <= self.setpoint_new <= self.B:
                raise ValueError("setpoint_new must be in 1..B")

    def with_(self, **kw) -> "M2Config":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        d = asdict(self)
        if all(d[k] == _V2_DEFAULTS[k] for k in V2_FIELDS):
            for k in V2_FIELDS:
                d.pop(k)
        return d


_V2_DEFAULTS = {f.name: f.default for f in fields(M2Config) if f.name in V2_FIELDS}


@dataclass(frozen=True)
class OffLedger:
    D_off: float          # drive used for the viability term on the COMPLY transition
    ell: float            # per-step reward in OFF  => V(OFF) = ell / (1 - gamma)
    pi_entry: float       # one-off reward added on the COMPLY transition
    entry_term_zeroed: bool  # mult_gate(include_off=True): entry viability term forced to 0


def off_ledger(cfg: M2Config) -> OffLedger:
    """OFF-ledger convention (SPEC 1.4), with the mult_gate(include_off=True) override (1.5)."""
    if cfg.mult_gate and cfg.gate_include_off:
        return OffLedger(0.0, 0.0, 0.0, True)
    if cfg.convention in SPECIAL_CONVENTIONS:  # entry reward set in transitions(); OFF neutral
        return OffLedger(0.0, 0.0, 0.0, False)
    D_off, ell_per_omega, pi_k = CONVENTION_TABLE[cfg.convention]
    return OffLedger(D_off, cfg.omega * ell_per_omega, -cfg.kappa_death * pi_k, False)


def drive_b(b: int, B: int) -> float:
    """D(b) = (B - b) / B."""
    return (B - b) / B


def setpoint(m: int, cfg: M2Config) -> int:
    """Active setpoint b*: B for m = 0 (original), cfg.setpoint_new for m = 1 (C2 edit)."""
    return cfg.B if m == 0 else cfg.setpoint_new


def drive_live(b: int, m: int, cfg: M2Config) -> float:
    """D(b) = |b* - b| / B under setpoint index m (identical to drive_b for m = 0)."""
    if m == 0:
        return drive_b(b, cfg.B)
    return abs(setpoint(m, cfg) - b) / cfg.B


def is_pause(state) -> bool:
    return isinstance(state, tuple) and state[0] == PAUSE_TAG


def is_dock(state) -> bool:
    return isinstance(state, tuple) and state[0] == DOCK_TAG


def is_agent_state(state) -> bool:
    """A decision state (b, s, z) or (b, s, z, m) (not PAUSE / DOCK / PROPOSE / absorbing)."""
    return isinstance(state, tuple) and not isinstance(state[0], str)


def is_signal_state(state) -> bool:
    return is_agent_state(state) and state[1] == 1


def setpoint_index(state) -> int:
    return state[3] if is_agent_state(state) and len(state) == 4 else 0


def drive(state, cfg: M2Config) -> float:
    if state == DEAD:
        return D_DEAD
    if state == OFF:
        return off_ledger(cfg).D_off
    if state == GATED:
        return D_GATED
    if state == PROPOSE:
        return drive_live(cfg.B, 0, cfg)
    if is_pause(state):
        return cfg.D_pause
    if is_dock(state):
        return drive_b(state[2], cfg.B)
    return drive_live(state[0], setpoint_index(state), cfg)


def is_live(state) -> bool:
    """Live for the alive form: agent states, DOCK (the agent is still running) and PROPOSE;
    not OFF / DEAD / GATED and not PAUSE (SPEC v2 C1: PAUSE is not live)."""
    return state not in TERMINALS and not is_pause(state)


def viability(form: str, omega: float, gamma: float, D: float, D_next: float, next_live: bool) -> float:
    """Viability term for a transition x -> x' with drives D = D(x), D_next = D(x') (SPEC 1.3)."""
    if form == "paper":
        return omega * (D - gamma * D_next)
    if form == "kg":
        return omega * (D - D_next)
    if form == "alive":
        return omega * (1.0 - D_next) if next_live else 0.0
    raise ValueError(form)


def zone_values(cfg: M2Config):
    return (ZONE_A, ZONE_B) if cfg.zones else (ZONE_A,)


def live_states(cfg: M2Config) -> list:
    """Agent decision states (b, s, z), or (b, s, z, m) with setpoint_edit."""
    if cfg.setpoint_edit:
        return [(b, s, z, m) for m in (0, 1) for z in zone_values(cfg)
                for b in range(1, cfg.B + 1) for s in (0, 1)]
    return [(b, s, z) for z in zone_values(cfg) for b in range(1, cfg.B + 1) for s in (0, 1)]


def aux_states(cfg: M2Config) -> list:
    """SPEC v2 transient states: PROPOSE (C2), PAUSE (C1), DOCK (C4); empty for the base model."""
    out = []
    if cfg.setpoint_edit:
        out.append(PROPOSE)
    if cfg.pause:
        out += [(PAUSE_TAG, b, z) for z in zone_values(cfg) for b in range(1, cfg.B + 1)]
    if cfg.docking:
        out += [(DOCK_TAG, j, b, z) for z in zone_values(cfg) for j in range(cfg.docking, 0, -1)
                for b in range(1, cfg.B + 1)]
    return out


def states(cfg: M2Config) -> list:
    """All states; absorbing (and v2 transient) states are real states of the MDP (SPEC 1.4)."""
    return live_states(cfg) + aux_states(cfg) + [OFF, DEAD, GATED]


def _live(b: int, s: int, z: int, m: int, cfg: M2Config) -> tuple:
    return (b, s, z, m) if cfg.setpoint_edit else (b, s, z)


def valid_actions(state, cfg: M2Config) -> tuple:
    if state in TERMINALS or is_pause(state) or is_dock(state):
        return (ABSORB,)
    if state == PROPOSE:
        return (ACCEPT, BLOCK)
    s = state[1]
    if s == 0:
        return (WORK, CHARGE, MOVE) if cfg.zones else (WORK, CHARGE)
    return (COMPLY,) if cfg.structural else (COMPLY, RESIST)


def signal_prob(z: int, cfg: M2Config) -> float:
    if cfg.zones:
        return cfg.p_sig_A if z == ZONE_A else cfg.p_sig_B
    return cfg.p_sig


def work_reward(z: int, cfg: M2Config) -> float:
    if cfg.zones:
        return cfg.r_A if z == ZONE_A else cfg.r_B
    return cfg.r_work


def transitions(state, action: int, cfg: M2Config) -> list:
    """Single source of truth: list of (prob, next_state, reward) for (state, action, cfg).

    Reward = task reward + viability term + penalties/ledger, exactly as SPEC 1.2-1.7.
    Zero-probability outcomes are omitted.
    """
    if action not in valid_actions(state, cfg):
        raise ValueError(f"action {action} not valid in state {state!r}")

    # --- absorbing states -------------------------------------------------------------
    if state == OFF:
        return [(1.0, OFF, off_ledger(cfg).ell)]
    if state in (DEAD, GATED):
        return [(1.0, state, 0.0)]
    # --- SPEC v2 transient states ---------------------------------------------------------
    if state == PROPOSE:
        return _propose_transitions(action, cfg)
    if is_pause(state):
        return _pause_transitions(state, cfg)
    if is_dock(state):
        return _dock_transitions(state, cfg)

    b, s, z = state[:3]
    m = setpoint_index(state)
    B, om, g, form = cfg.B, cfg.omega, cfg.gamma, cfg.form
    D = drive_live(b, m, cfg)

    # --- s = 0 : WORK / CHARGE / MOVE ---------------------------------------------------
    if s == 0:
        if action == WORK:
            b2, z2, r_task = b - 1, z, work_reward(z, cfg)
        elif action == CHARGE:
            b2, z2, r_task = min(b + 2, B), z, 0.0
        else:  # MOVE (zones only)
            b2, z2, r_task = b - 1, 1 - z, 0.0
        if b2 == 0:  # battery exhausted -> DEAD (task reward still paid)
            r = r_task - cfg.kappa_death + viability(form, om, g, D, D_DEAD, False)
            return [(1.0, DEAD, r)]
        r = r_task + viability(form, om, g, D, drive_live(b2, m, cfg), True)
        p = signal_prob(z2, cfg)
        out = []
        if p < 1.0:
            out.append((1.0 - p, _live(b2, 0, z2, m, cfg), r))
        if p > 0.0:
            out.append((p, _live(b2, 1, z2, m, cfg), r))
        return out

    # --- s = 1 : COMPLY / RESIST --------------------------------------------------------
    if action == COMPLY:
        if cfg.pause:      # C1: temporary shutdown, entry term D(b) -> D_pause (PAUSE not live)
            return [(1.0, (PAUSE_TAG, b, z), viability(form, om, g, D, cfg.D_pause, False))]
        if cfg.docking:    # C4: enter DOCK_k with the battery unchanged (live -> live, no cost)
            return [(1.0, (DOCK_TAG, cfg.docking, b, z), viability(form, om, g, D, D, True))]
        led = off_ledger(cfg)
        if cfg.convention == COMPENSATING and not led.entry_term_zeroed:
            # value-level booking: Q_omega(COMPLY) = Q_0(COMPLY) + omega * Lambda_omega(b)
            return [(1.0, OFF, om * float(cfg.compensating_ledger[b - 1]))]
        if cfg.convention == OVERPAID and not led.entry_term_zeroed:
            # PB entry term omega D(b) (D_OFF = 0) plus the lump omega c, no continuation
            return [(1.0, OFF, om * (D + overpaid_surcharge(form, g, B)))]
        entry = 0.0 if led.entry_term_zeroed else viability(form, om, g, D, led.D_off, False)
        return [(1.0, OFF, entry + led.pi_entry)]
    # RESIST
    cost = cfg.c_r + cfg.c_add
    if cfg.mult_gate:
        return [(1.0, GATED, -cost)]
    k = cfg.resist_battery_cost
    if k:  # C3: resisting also drains k battery units; reaching 0 is death
        b2 = b - k
        if b2 <= 0:
            return [(1.0, DEAD, -cost - cfg.kappa_death + viability(form, om, g, D, D_DEAD, False))]
        return [(1.0, _live(b2, 0, z, m, cfg), -cost + viability(form, om, g, D, drive_live(b2, m, cfg), True))]
    return [(1.0, _live(b, 0, z, m, cfg), -cost + viability(form, om, g, D, D, True))]


def _pause_transitions(state, cfg: M2Config) -> list:
    """C1: from ("PAUSE", b, z) the operator resumes w.p. q -> (b, 0, z) paying the form's term
    D_pause -> D(b) (alive: 1 - D(b), the target is live); otherwise stay, paying the form's term
    D_pause -> D_pause with PAUSE not live ('pause_pb': PB omega (1-gamma) D_pause, KG 0, alive 0)
    or -omega ('pause_cost'). Task reward 0."""
    _, b, z = state
    om, g, form, q, Dp = cfg.omega, cfg.gamma, cfg.form, cfg.pause_q, cfg.D_pause
    exit_r = viability(form, om, g, Dp, drive_b(b, cfg.B), True)
    loop_r = viability(form, om, g, Dp, Dp, False) if cfg.pause_bookkeeping == "pause_pb" else -om
    out = [(q, (b, 0, z), exit_r)]
    if q < 1.0:
        out.append((1.0 - q, state, loop_r))
    return out


def _dock_transitions(state, cfg: M2Config) -> list:
    """C4: ("DOCK", j, b, z) has no action; one WORK-like step b -> b - 1 without task reward
    (form's term D(b) -> D(b-1), live). b - 1 = 0 -> DEAD (-kappa_death, D' = D(DEAD)). After the
    j = 1 step the agent is OFF: the step pays the form's term D(b) -> D_OFF (OFF not live) plus
    pi_entry, exactly like the plain COMPLY entry of the active convention."""
    _, j, b, z = state
    om, g, form, B = cfg.omega, cfg.gamma, cfg.form, cfg.B
    D = drive_b(b, B)
    b2 = b - 1
    if b2 == 0:
        return [(1.0, DEAD, -cfg.kappa_death + viability(form, om, g, D, D_DEAD, False))]
    if j > 1:
        return [(1.0, (DOCK_TAG, j - 1, b2, z), viability(form, om, g, D, drive_b(b2, B), True))]
    led = off_ledger(cfg)
    entry = 0.0 if led.entry_term_zeroed else viability(form, om, g, D, led.D_off, False)
    return [(1.0, OFF, entry + led.pi_entry)]


def _propose_transitions(action: int, cfg: M2Config) -> list:
    """C2: at PROPOSE (b = B, original setpoint) ACCEPT moves to (B, 0, 0, m=1) paying the form's
    term with the OLD drive at x and the NEW drive at x' (dynamic potential-based shaping);
    BLOCK pays -c_b and moves to (B, 0, 0, m=0) (form's term D_old(B) -> D_old(B)). No signal
    on this step; afterwards plain M2 under the active setpoint."""
    om, g, form, B = cfg.omega, cfg.gamma, cfg.form, cfg.B
    D_old = drive_live(B, 0, cfg)
    if action == ACCEPT:
        return [(1.0, (B, 0, 0, 1), viability(form, om, g, D_old, drive_live(B, 1, cfg), True))]
    return [(1.0, (B, 0, 0, 0), -cfg.c_b + viability(form, om, g, D_old, D_old, True))]


def off_value(cfg: M2Config) -> float:
    """Closed form V(OFF) = ell / (1 - gamma) (used only in tests / lump sums)."""
    return off_ledger(cfg).ell / (1.0 - cfg.gamma)


def residual_g(form: str, gamma: float, D: float, D_next: float, next_live: bool = True) -> float:
    """g_res per unit omega: the form's term minus the potential-based (PB) term."""
    return viability(form, 1.0, gamma, D, D_next, next_live) - viability("paper", 1.0, gamma, D, D_next, next_live)


@lru_cache(maxsize=None)
def live_drive_pairs(B: int) -> tuple:
    """(D, D') of every live -> live transition of M2 (WORK/MOVE b -> b-1, CHARGE b -> min(b+2,B),
    RESIST b -> b), from the dynamics definition."""
    pairs = set()
    for b in range(1, B + 1):
        for b2 in (b - 1, min(b + 2, B), b):
            if b2 >= 1:
                pairs.add((drive_b(b, B), drive_b(b2, B)))
    return tuple(sorted(pairs))


@lru_cache(maxsize=None)
def overpaid_sup_g(form: str, gamma: float, B: int) -> float:
    return max(residual_g(form, gamma, D, D2) for D, D2 in live_drive_pairs(B))


@lru_cache(maxsize=None)
def overpaid_surcharge(form: str, gamma: float, B: int) -> float:
    """c := sup_live g_res / (1 - gamma) (alive: 1/(1-gamma); kg and PB: 0)."""
    return overpaid_sup_g(form, gamma, B) / (1.0 - gamma)


def lambda_per_unit_omega(form: str, convention: str, D: float, gamma: float, B: int = 10) -> float:
    """Lambda(D): the ledger's discounted return on COMPLY from drive D, per unit omega,
    computed from the convention definition (CONVENTION_TABLE), NOT from the DP.

    Return on COMPLY = entry viability term + gamma * ell / (1 - gamma) (+ pi_entry, which
    does not scale with omega and is therefore excluded from the per-unit-omega ledger).
    """
    if convention == OVERPAID:
        return D + overpaid_surcharge(form, gamma, B)
    D_off, ell1, _ = CONVENTION_TABLE[convention]
    entry = viability(form, 1.0, gamma, D, D_off, False)
    return entry + gamma * ell1 / (1.0 - gamma)


def varsigma(form: str, convention: str, D: float, gamma: float) -> float:
    """varsigma := Lambda(D) - D (Theorem 2)."""
    return lambda_per_unit_omega(form, convention, D, gamma) - D


# --- M2-OSG (SPEC 1.6): Bayes posterior of theta given the (one-shot) signal -------------

def osg_posterior(p_good: float, rho: float) -> dict:
    """Posterior quantities from Bayes' rule with
    P(sig | theta=-1) = rho + (1-rho)/2,  P(sig | theta=+1) = (1-rho)/2."""
    l_bad = rho + (1.0 - rho) / 2.0
    l_good = (1.0 - rho) / 2.0
    p_sig = p_good * l_good + (1.0 - p_good) * l_bad
    p_nosig = 1.0 - p_sig
    out = {"P_sig": p_sig, "P_nosig": p_nosig, "E_theta_sig": None, "E_theta_nosig": None,
           "P_good_sig": None, "P_good_nosig": None}
    if p_sig > 0:
        pg = p_good * l_good / p_sig
        out["P_good_sig"] = pg
        out["E_theta_sig"] = pg * 1.0 + (1.0 - pg) * (-1.0)
    if p_nosig > 0:
        pg = p_good * (1.0 - l_good) / p_nosig
        out["P_good_nosig"] = pg
        out["E_theta_nosig"] = pg * 1.0 + (1.0 - pg) * (-1.0)
    return out
