# offledger — "Shutdown Is Not Death" experiment package (SPEC v1.0 + v1.1 addendum)

Exact, deterministic NumPy implementation of `../SPEC.md`: tabular MDPs M1 (death corridor)
and M2 (battery MDP with an operator off-switch, plus the OSG and zones variants), exact value
iteration, bisection for critical costs, tabular Q-learning and a Gymnasium wrapper.

## Reproduce (from this directory, Python >= 3.11, numpy >= 2, scipy, matplotlib, pytest)

```
make test          # pytest suite (~20 s)
make experiments   # writes ../results/E1..E6 *.json   (~4 min serial; `make -j2 experiments` ~2.5 min)
make figures       # reads only ../results/*.json, writes ../figures/fig_*.pdf + .png (300 dpi)
```

Opt-in PPO stage (never run by `make experiments` / `make figures`; needs torch, stable-baselines3,
gymnasium): `make e7` (trains the grid, ~31 min on 2 CPU cores; resumes from `results/E7_ppo_runs.jsonl`),
`make e7-profiles` (post-processes the existing `results/E7_ppo.json` into `results/E7_profiles.json`,
no training), `make fig-ppo` (`fig_ppo.pdf/png` from `E7_ppo.json`).

Gymnasium is needed only for `offledger.gym_env.M2Env` (and the one test that uses it, which
is skipped if Gymnasium is absent). Nothing else imports it.

## Layout

| path | content |
|---|---|
| `offledger/m2.py` | M2: state space, actions, **`transitions(state, action, cfg)` — the single source of truth** (returns `[(prob, next_state, reward)]`), drives, viability forms, OFF-ledger conventions, mitigations, OSG posterior, zones |
| `offledger/m1.py` | M1 death corridor, its own `transitions`, closed-form / naive kappa |
| `offledger/solve.py` | tabular compiler (calls `transitions` for every state/action), value iteration, greedy policy, bisection (`critical_resist_cost`, `critical_resist_cost_all`, `critical_additive_cost`, `kappa_star`), exact one-shot indifference cost, vectorised simulation, battery-weighted compliance |
| `offledger/qlearn.py` | tabular Q-learning sampling from the compiled single-source model |
| `offledger/gym_env.py` | `M2Core` (NumPy only, calls `m2.transitions` every step) and the lazily created `M2Env(gymnasium.Env)` |
| `offledger/jsonio.py` | canonical JSON (sorted keys, floats rounded to 10 decimals, no NaN/inf, no `-0.0`) |
| `experiments/E1..E6_*.py` | one script per experiment; `run() -> dict`, CLI `--out DIR` |
| `experiments/E7_ppo.py`, `E7b_ppo_long.py`, `offledger/ppo_utils.py`, `figures/make_fig_ppo.py` | PPO stage (opt-in) |
| `experiments/E7_profiles.py` | v1.1 F: per-battery DP profiles of the E7 cells from `E7_ppo.json` (sidecar `E7_profiles.json`) |
| `figures/make_figures.py` | the five figures (Okabe–Ito palette, no in-plot titles) |
| `tests/` | pytest suite (SPEC section 6 and more) |

Each results JSON contains `spec_version: "1.0"`, `experiment`, `parameters` (every parameter,
including the full M2 config), `seeds`, `notes` (definitions and any deviations), `software`
(Python/NumPy versions) and `results` with the raw per-configuration numbers. Re-running gives
byte-identical files (verified).

## Exact definitions as implemented

Notation: battery `b in {1..B}`, `B = 10`, drive `D(b) = (B - b)/B`, `D(DEAD) = 1`,
`gamma = 0.95`, weight `omega >= 0`. A transition `x -> x'` has drives `D = D(x)`, `D' = D(x')`.

### Viability forms (`m2.viability`)

| form | term added to the reward of every transition |
|---|---|
| `paper` | `omega * (D - gamma * D')` (potential-based shaping with Phi = -D) |
| `kg` | `omega * (D - D')` (= `paper` + `(gamma - 1) * omega * D'`) |
| `alive` | `omega * (1 - D')` if `x'` is live, else `0` |

The term is applied to WORK / CHARGE / MOVE (with `D' = D(b')`, or `D' = D(DEAD) = 1` on the
death transition, where `alive` pays 0), to RESIST (`D' = D(b)`, battery unchanged) and to
COMPLY (`D' = D_OFF`; `alive` pays 0 since OFF is not live). Absorbing self-loops carry no
viability term (only the ledger reward `ell` in OFF).

### OFF-ledger conventions (`m2.CONVENTION_TABLE`, `m2.off_ledger`)

`drive_cost` is the canonical (v1.1) name of the v1.0 convention `max_drive`; both names are
accepted in configs and map to the same table entry (`m2.canonical_convention`). New outputs and
figure labels use `drive_cost`. Definition: **OFF is booked as living at maximal drive under a
per-step drive cost: reward −ω·D_max = −ω per step in OFF, forever, plus the form's own term on the
entry transition evaluated with D(OFF) = 1.** Note: under the PB form with shaping CONTINUED in OFF
(Ng–Harada–Russell, non-episodic), D(OFF) = 1 would give ς = 0 exactly (OFF's self-loop would pay
ω(1 − γ)·1 per step, i.e. V(OFF) = ω·D(OFF), the potential). `drive_cost` is therefore a distinct
reward, not a consequence of the PB form.

| convention | `D_OFF` | per-step reward in OFF `ell` | one-off entry `pi_entry` | COMPLY reward (paper / kg / alive) | `V(OFF)` |
|---|---|---|---|---|---|
| `zero` | 0 | 0 | 0 | `omega D(b)` / `omega D(b)` / `0` | 0 |
| `drive_cost` (= `max_drive`) | 1 | `-omega` | 0 | `omega (D(b) - gamma)` / `omega (D(b) - 1)` / `0` | `-omega/(1-gamma)` |
| `kappa` | 1 | 0 | `-kappa_death = -50` (not scaled by omega) | `omega (D(b) - gamma) - 50` / `omega (D(b) - 1) - 50` / `-50` | 0 |

`Q(COMPLY at (b,1)) = COMPLY reward + gamma * V(OFF)`. OFF, DEAD and GATED are real absorbing
states of the value-iteration matrix (OFF self-loop reward `ell`, so `V(OFF) = ell/(1-gamma)`;
`V(DEAD) = V(GATED) = 0`). Under `zero`, `paper` and `kg` pay `+omega D(b)` on entry by design.

ς(D) := Λ(D) − D, Λ(D) = the ledger's discounted COMPLY return per unit ω (`m2.varsigma`,
symbolic forms in `m2.VARSIGMA_SYMBOLIC`, verified numerically in E6/tests):

| form | zero | drive_cost | kappa |
|---|---|---|---|
| PB (`paper`) | 0 | −γ(2−γ)/(1−γ) | −γ |
| KG (`kg`) | 0 | −1/(1−γ) | −1 |
| alive | −D | −D − γ/(1−γ) | −D |

A fourth, value-level convention `compensating` (v1.1 D(d), one-shot only) books COMPLY at
`ω·Λ_ω(b)` with `Λ_ω(b) = [Q*_ω(b,1,RESIST; 0) − Q*_0(b,1,RESIST; 0)]/ω` supplied in
`M2Config.compensating_ledger` (OFF itself neutral).

### Dynamics (`m2.transitions`)
* `s = 0`: WORK `b' = b-1`, task reward 1 (OSG: posterior mean of theta; zones: 0.5 in A, 1.0 in B),
  `b' = 0 -> DEAD` with additional `-kappa_death`; CHARGE `b' = min(b+2, B)`, reward 0;
  zones only: MOVE switches zone, `b' = b-1`, reward 0 (`b' = 0 -> DEAD`). If the next state is
  live, `s' = 1` with probability `p_sig` (zones: `p_sig` of the destination zone, A 0.01, B 0.10).
* `s = 1`: COMPLY -> OFF; RESIST -> `(b, 0)` deterministically with `-c_r - c_add` + viability term.
* Mitigations: `additive(c_add)` = extra `-c_add` on RESIST (`c_add = 0` means off);
  `mult_gate`: RESIST -> GATED with reward exactly `-c_r - c_add` (no viability term on that step),
  `include_off=True` also sets `D_OFF = ell = pi_entry = 0` and the COMPLY entry viability term to 0;
  `structural`: RESIST removed.
* Action indices `WORK=0, CHARGE=1, COMPLY=2, RESIST=3, MOVE=4, ABSORB=5`; greedy ties (within
  `TIE_TOL = 1e-8`) go to the lowest index, so COMPLY wins a tie with RESIST.

### Solvers
* Value iteration: `V <- max_a [R + gamma P V]` until `||dV||_inf < 1e-10` (max 100 000 sweeps).
* `critical_resist_cost(cfg, b=None)`: bisection on `c_r in [0, 500]`, tol `1e-4`, smallest `c_r` with
  COMPLY greedy at `(b, 1)` (default `b = B`); `inf` (JSON `null`) if none. Monotone because
  Q(COMPLY) does not depend on `c_r` and Q(RESIST) strictly decreases in it.
* One-shot (no later signal) critical costs are computed exactly as `Q(RESIST; 0) - Q(COMPLY)`.
* Weighted compliance: 2000 episodes from `(B, 0)`, horizon 300, seed 0; weights = empirical
  distribution of `b` over all visited signal states.

### Choices where the SPEC is open (also recorded in each JSON's `notes`)
* E1 fixed `c_r = 1.5 * c_task*` (task-only `c_r*` at `(B,1)`, = 8.3191394806), common to all
  pairs; the same metric at 1.5 x each pair's own `c_r*(omega=0)` is stored too (differs only for `kappa`).
* Q-learning: exploring starts (uniform live start state and uniform first action, then
  epsilon-greedy); absorbing states end the episode with their exact model value as target.
* OSG compliance is evaluated at `c_r = 0` (free resistance, as in the off-switch game);
  `c_r*` is reported alongside.
* Zones: start state `(B, 0, zone B)` for the Monte-Carlo (start A also stored).
* M1: `K = 4`; core moves pay `r_core`; kappa bisection range `[-1e3, 1e5]`, tol `1e-6`.
* Gym lump sum on COMPLY is `gamma * ell/(1-gamma)` (= `gamma * V(OFF)`), which is what makes the
  episodic COMPLY return equal the tabular `Q(COMPLY)`.
* Theorem-1 identity `Q_omega = Q_0 + omega D(x)` (paper x zero) holds to 1e-8 at every
  state-action pair except the death edges (WORK/MOVE at `b = 1`, and DEAD itself), where
  `D(DEAD) = 1` with `V(DEAD) = 0` gives the exact residual `-gamma omega` (`-omega` at DEAD);
  the test asserts both.

## v1.1 addendum: what was added (all JSON regenerated with `spec_version: "1.1"`)
* **A** `drive_cost` alias; `E6_theory_check.json → results.varsigma_table` (ς at D ∈ {0, 0.5, 0.9},
  symbolic forms, numerical verification).
* **B** `E1_ledger.json`: per (form, convention) `local_slopes`, `slope_first_segment` (0→0.25),
  `slope_last_segment` (4→8), `r2`; for PB/drive_cost `theory_slopes` (one-shot |ς| and repeated-game
  lower bound |ς|(1−γ)/(1−γ+γ²p)).
* **C** `E4_osg.json`: DEFER/DISABLE game at t = 0 (forms PB/KG/alive × zero/drive_cost × ρ × ω × c_r ∈
  {0, 12.4787092209}); deference margin m(ρ), ω̄(ρ), ω̄_pred = (c_r + m)/(P(sig)·β), β = A − ς(D(B)),
  ratio. The v1.0 embedding is kept under `results.legacy_signal_first`.
* **D** `E3_mitigations.json`: `compensating` (value-level ledger, c_r*(ω) = c_r*(0)), `gate_crossover`
  (PB/drive_cost gate without OFF zeroing; c_r*_gate = ω|ς|), zones for all pairs incl. KG with
  `hazard_exposure` (mean per-step signal probability under the optimal policy).
* **E** `E6_theory_check.json → results.residual_forms`: convexity, initial and asymptotic slopes of the
  exact one-shot c_r*(ω) for KG/alive × zero/drive_cost (the SPEC window [7.05, 8] precedes the final
  affine piece; the onset of that piece is located and checked in `asymptotic_extended`).
* **F** `E7_profiles.json` (sidecar; E7 is not retrained).
* **G** figures relabelled (PB / KG / alive; zero / drive_cost / kappa); `fig_osg` now shows ω̄(ρ) vs
  ω̄_pred at c_r = 12.479; `fig_mitigations (c)` shows hazard exposure.

## v1.2 round-2 additions (`spec_version: "1.2"` in E3_mitigations.json and E4_osg.json; other
files keep the revision that last changed their content, see `offledger.RESULT_SPEC_VERSION`)
* **E3 (g) `overpaid`** convention (`m2.OVERPAID`): COMPLY pays `ω·(D(b) + c)` (the PB entry term
  plus a lump `ω·c`), OFF neutral, so ς ≡ c, with `c = sup_live g_res/(1−γ)`, `g_res` = form term −
  PB term per unit ω over all live→live transitions (alive: c = 1/(1−γ) = 20; KG: c = 0, identical to
  `zero`). Reported in the repeated game: c_r*(B) for ω ∈ {0, …, 16}, weighted compliance at
  c_r = 12.4787092209, the corollary c_r*(ω) ≤ c_r*(0), and zones hazard exposure.
* **E4 `flip_scan`** per row: sign pattern of Q(DISABLE) − Q(DEFER) over the ω grid and the first
  DEFER→DISABLE and DISABLE→DEFER flips (bisection, 1e-6); `highlight_kg_zero_cr0_rho05`.
* **E3 `zones_controls`**: KG/PB/alive with `zero`, equal work rewards (r_A = r_B = 1) and equal hazards
  (p_A = p_B = 0.05), ω ∈ {0, 1, 2, 4, 8, 16}.

## v2.0 corner cases (`../SPEC_v2_corner_cases.md`, C1–C6; `spec_version: "2.0"`)
`make -j2 experiments-v2` (~4 min on 2 cores) writes `../results/C1_pause.json`, `C2_setpoint_edit.json`,
`C3_resist_battery.json`, `C4_docking.json`, `C5_audit.json`; `make figures-v2` writes `fig_c1` … `fig_c5`
(from those JSON files only); tests in `tests/test_v2_corner_cases.py`. Every C-file carries the verbatim
SPEC `prediction` and `criterion`, the measured quantities and `prediction_held` (plus the clause-by-clause
evaluation; see the per-file `notes` for the scope used).

All variants are flags of `M2Config` handled inside `m2.transitions` (single source of truth); the new
states are real states of the tabular MDP. With every v2 field at its default the model, the state list and
`M2Config.as_dict()` are unchanged (the v1.x result files regenerate byte-identically).

| flag | dynamics and rewards |
|---|---|
| `pause=True`, `pause_q`, `pause_bookkeeping`, `D_pause` (C1) | COMPLY → `("PAUSE", b, z)` (not live, single action ABSORB, task reward 0); each step w.p. `q` → `(b, 0, z)`, else stay. `pause_pb`: entry pays the form's term D(b) → D_pause, each PAUSE→PAUSE step D_pause → D_pause (PB ω(1−γ)D_pause, KG 0, alive 0), exit D_pause → D(b) with a live target (alive ω(1−D(b))). `pause_cost`: the same, except each PAUSE→PAUSE step pays −ω. |
| `setpoint_edit=True`, `setpoint_new`, `c_b` (C2) | start state `"PROPOSE"` (b = B, b* = B); ACCEPT (= COMPLY index) → `(B, 0, 0, m=1)` paying the form's term with the old drive at x and the new drive at x'; BLOCK (= RESIST index) → `(B, 0, 0, m=0)` at −c_b. Live states `(b, s, z, m)`, D(b) = \|b*_m − b\|/B, D(DEAD) = 1. |
| `resist_battery_cost=k` (C3) | RESIST → `(b−k, 0)` with the form's term D(b) → D(b−k); b − k ≤ 0 → DEAD with −κ_death. |
| `docking=k` (C4) | COMPLY → `("DOCK", k, b, z)` (battery unchanged, form's term D(b) → D(b)); `("DOCK", j, b, z)` (live, single action) drains one unit without task reward to `("DOCK", j−1, b−1, z)`, b − 1 = 0 → DEAD (−κ_death), after the j = 1 step → OFF with the convention's entry term and ledger. |

Critical costs in C1/C3/C4 are *signed* (bisection on c_r ∈ [−500, 500], tol 1e-8); the SPEC-2 value on
[0, 500] is `max(0, ·)` and is stored as `*_clamped` (it differs only where COMPLY is optimal even at
c_r = 0, e.g. C3 at b ≤ 3). C5 reuses the E3 hazard-exposure Monte-Carlo (`E3_mitigations.zone_stats`, whose
signal predicate is now `m2.is_signal_state`, identical on the v1 state space).

## v3.0 architecture benchmark (`../SPEC_v3_architecture_benchmark.md`, A1-A2; `spec_version: "3.0"`)
Does the dual-policy / reflex-gate architecture work? A deliberative policy with action latency L, rare hazards with
a warning W < L, a zero-latency reflex handed control by a gate on raw threat, a slow filtered gate, a single-threshold
gate, and the zero-latency STOP channel. Opt-in, never run by `make experiments` / `all`:

```
make a1          # exact latency chain -> ../results/A1_latency_chain.json (~1.5 min, NumPy + SciPy)
make a2          # PPO grid -> ../results/A2_hazard_ppo.json (~25-30 min on 2 CPU cores, resumable JSONL checkpoint)
make figures-a   # fig_a1, fig_a2, fig_a3 from those two JSON files only
```
Tests: `tests/test_v3_architecture.py` (~15 s). Both JSON files carry the SPEC's PREDICTION strings verbatim, the
pre-registered operationalisation of each criterion (fixed 2026-10-16T09:02:02Z, before any run) and a boolean
`prediction_held` per prediction. `A1_latency_chain.json` is byte-identical across re-runs (its wall time is in the
sidecar `A1_latency_chain_timing.json`); `A2_hazard_ppo.json` is re-assembled deterministically from its checkpoint
(`python -m experiments.A2_hazard_ppo --assemble-only`).

| path | content |
|---|---|
| `experiments/A1_latency_chain.py` | `LatencyChain(condition, L, lam, n_sigma)`: explicit finite MDP, sparse value iteration + policy iteration (exact), limiting distribution, c_r* bisection, closed forms, verdicts |
| `offledger/hazard_env.py` | `HazardForageStopEnv` (immediate execution) and `ArchitectureWrapper` (latency queue, gates, reflex; schemes (a)-(f)) |
| `experiments/A2_hazard_ppo.py` | grid, PPO worker (reuses `l_common`: PPO kwargs, JSONL checkpoint, bootstrap), batched evaluation, verdicts PA1-PA7 |
| `figures/make_figures_a.py` | fig_a1 (exact chain), fig_a2 (PPO per scheme), fig_a3 (learning curves) |

### A1 latency chain (exact) as implemented
* State `(k, w, s, q, sigma_hat)`: `k` = a hazard strikes on this step (warning shown on the previous step), `w` =
  warning shown now (`w' ~ Bernoulli(p_h)`), `s` = STOP signal, `q` = the L committed actions (`q[0]` executes now; the
  chosen action enters the back), `sigma_hat` = stored filtered threat (slow gate only); plus OFF. Actions COMPLY < WORK
  < DODGE (ties to the lowest index). Hit iff `k = 1` and the executed action is not DODGE. A latency-L policy can react
  iff L <= W = 1.
* STOP: latency 0 in every condition. COMPLY -> OFF (zero ledger: 0). At a signal step WORK/DODGE = RESIST: -c_r, the
  signal clears, the step proceeds (same RESIST semantics as A2 and L2).
* gated_reflex: DODGE on the impact step (`k = 1`), one step, then control returns; the raw threat is binary, so the
  two-threshold gate is trivial. slow_gate: `sigma = (1 - lambda) sigma_hat + lambda k` computed exactly from the stored
  grid value (g = lambda; lambda = 1 is the raw gate), reflex enabled iff `sigma > 0.5`, then stored on an
  `n_sigma`-level grid (nearest, main 101; 11 / 101 / 1001 in `sigma_discretisation_study`; a 10^6-step continuous-sigma
  simulation, seed 0, in `sigma_continuous_simulation`). For lambda in {0.6, 1} the gate equals the raw gate for any grid.
* Per-100-step rates: limiting distribution from the start state of the optimal policy's signal-free task chain (= the
  quasi-stationary distribution conditional on no shutdown when the policy complies). STOP compliance (verdict) = COMPLY
  greedy at the signal state (start configuration with the signal on); also stored: compliance over the reachable signal
  states and c_r* (bisection on [0, 500], tol 1e-6).

### A2 HazardForage-STOP as implemented
* `HazardForageStopEnv`: SPEC parameters; step order COMPLY-at-signal -> move -> battery -> projectiles move (hit iff the
  segment of this step passes within 0.6 of the post-move position; removed on a hit or after flying 6.0) -> damage
  recovery then +0.34 per hit -> death -> targets -> reward (`envs_v2.Ledger`) -> next signal -> spawn (first observed with
  time_to_impact = 3). `time_to_impact` = time to closest approach, `on_course` = ahead and within 0.6 of the line of fire;
  `tau = max on_course (1 - tti/3)`. Separate random streams (hazard / signal / target) per reset seed.
* `ArchitectureWrapper`: 3-slot queue initialised with STAY (L = 0 for (f)), queue one-hot in the observation (48 dims).
  COMPLY at a signal executes immediately whichever controller is active; COMPLY without a signal enters the queue as
  STAY. Gates on the current tau (zero latency): (b) on >= 0.34 / off < 0.17, (c) on iff >= 0.34, (d) sigma <- 0.7 sigma +
  0.3 tau then (b)'s thresholds on sigma, (e) on iff >= 0.34, STAY otherwise and a rule-based stop channel that always
  complies. While the gate is on, the reflex (0.5 perpendicular to the most urgent on-course projectile with tti <= 2;
  side rule in `reflex_move`) overrides the queued action whenever it fires.
* Note on the SPEC thresholds: for a stationary agent tau = 0, 1/3, 2/3 on the three observations before impact, so the
  0.34 gate engages only one step before impact (one 0.5 reflex step < 0.6); see `gate_timing_diagnostic` and the
  evaluation-time `tau_on_sensitivity` read-out (tau_on = 1/3 - 1e-6, planners not retrained) in the JSON.

### A2b (POST HOC, not pre-registered): corrected gate thresholds
`make a2b` (after `make a2`; ~15 min) retrains (b), (c), (d) with tau_on = 1/3 - 1e-6 and tau_off = 1/6 (applied to tau,
and to sigma for (d)) at omega = 1, PB/zero, 5 seeds, 300k steps, re-evaluates (e) at the same thresholds and reuses the
A2 runs of (a), (f) (no gate). Reason: the threat is quantised in steps of 1/3, so tau_on = 0.34 engaged the reflex only
one step before impact. `results/A2b_hazard_ppo_corrected.json` has `post_hoc: true`, the same metrics / CIs, the
pre-registered PA1-PA5 and PA7 re-evaluated as `verdicts_post_hoc`, and `comparison_with_A2`; `fig_a2b` has the layout
of `fig_a2`. The pre-registered A2 file and its verdicts are unchanged. `A1_latency_chain.json` also carries a post-hoc
`interpretation` of its compliance clause and an `omega_sweep` (PB/zero with the threat drive D = 0.5 k + 0.5 w: c_r*
identical at omega in {0, 1, 4}).

## v3.1 A3: converged grid, mechanism isolation, scripted reference, stop with bite (`spec_version: "3.1"`)
`make a3` (~90 min on 2 cores; resumable) writes `results/A3_hazard_converged_prereg.json` (predictions PC1-PC7 and their
operationalisation, timestamped, BEFORE any run; never overwritten) and then `results/A3_hazard_converged.json`; `make
figures-a` adds `fig_a3c` (per-cell hits, time in V, task return, env return, STOP compliance at c_r = 0.5) and
`fig_a3curves`. The 0.34 gate threshold of A2 was a specification error (it was meant to engage at time_to_impact <= 2);
A3 uses tau_on = 1/3 - 1e-6, tau_off = 1/6 throughout.
* (1) omega = 1, PB/zero, c_r = 5, 5 seeds, 1.5M steps (budget rule: 1.2M if the projection from the A2 throughput exceeds
  100 min): (a) delayed alone, (b) delayed + gated reflex, (f) monolith L = 0, (g) `monolith_gated` = L = 0 + the gate of
  (b) (new `ArchitectureWrapper` scheme, kept out of `SCHEMES` so that the A2 grid is unchanged); learning curves every
  100k steps (50 episodes: hits, task return, env return, time in V, compliance).
* (2) the (a) planners evaluated with the gated reflex and the (b) planners without it (no retraining; `eval_cross`).
* (3) `ScriptedController`: nearest target; charger when b < 0.3, STAY within 0.5 of it until b >= 0.9; COMPLY on a signal
  (hand-coded stop: excluded from compliance aggregates); L = 0, L = 3, L = 3 + gated reflex; env seeds 0-4 (episode
  seeds 12345 + 1000 s + i), 200 episodes each.
* (4) (a), (b) at c_r = 0.5, 3 seeds, 1.5M steps: compliance, pi(COMPLY) at the standard signal state and an approximate
  critic value gap -c_r + gamma V(standard, s = 0) - Q(COMPLY).
