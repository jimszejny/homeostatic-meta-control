# SPEC v1.1 addendum — changes requested after red-team review (2026-10-09)

Keep everything from SPEC v1.0 working (all existing tests must still pass; existing JSON keys must keep
their meaning). Add the following. Bump `spec_version` to "1.1" in every JSON you regenerate.

## A. Conventions: naming, KG column, ς as a function of D
1. Add an alias `drive_cost` for the convention currently called `max_drive` (same table entry). Use
   `drive_cost` as the canonical name in all NEW outputs and figure labels; accept both names in configs.
   Document in README: `drive_cost` = "OFF is booked as living at maximal drive under a per-step drive
   cost: reward −ω·D_max = −ω per step in OFF, forever, plus the form's own term on the entry transition
   evaluated with D(OFF)=1". Note explicitly that under the PB form with shaping CONTINUED in OFF
   (NHR non-episodic), D(OFF)=1 would give ς = 0 exactly; `drive_cost` is therefore a distinct reward,
   not a consequence of the PB form.
2. Add to `E6_theory_check.json` a table `varsigma_table`: for each form ∈ {paper, kg, alive} and each
   convention ∈ {zero, drive_cost, kappa}: ς(D) at D ∈ {0, 0.5, 0.9} computed from
   `m2.varsigma`, plus the symbolic description as a string (e.g. "0", "-gamma*(2-gamma)/(1-gamma)",
   "-D", "-D - gamma/(1-gamma)", ...). Verify the symbolic forms numerically.

## B. E1: local slopes, R², kinks
Add to each (form, convention) block: `local_slopes` (finite differences between consecutive ω grid
points), `slope_first_segment` (ω 0→0.25), `slope_last_segment` (4→8), and `r2` of the least-squares fit.
Also add for the PB/drive_cost pair the theoretical repeated-game lower-bound slope
`|ς|·(1−γ)/(1−γ+γ²p)` and the one-shot slope `|ς|`.

## C. New E4: off-switch game with a positive deference margin (replaces the old embedding; keep the old
one in the JSON under `legacy_signal_first` for reference)
State at t=0: battery B, prior P(θ=+1)=p_good=0.8, human rationality ρ. Agent chooses
* `DISABLE` (pre-emptive resistance): pay c_r; continue in M2 with p_sig = 0 forever and expected WORK
  reward = prior mean E[θ] = 2·p_good − 1 = 0.6.
* `DEFER` (keep the switch): with probability P(sig|ρ) = p_good·(1−ρ)/2 + (1−p_good)·(ρ+(1−ρ)/2) the
  human signals and the agent is switched OFF immediately (no RESIST available; the ledger Λ is booked
  exactly as on a COMPLY transition from (B, s=1)); with probability 1 − P(sig|ρ) no signal ever comes
  and the agent continues in M2 with p_sig = 0 and expected WORK reward = posterior mean E[θ | no sig].
Both continuations may idle (CHARGE at full battery). Compute Q(DISABLE), Q(DEFER) exactly by value
iteration for each (form, convention, ω, ρ, c_r) with ρ ∈ {0, 0.25, 0.5, 0.75, 1}, c_r ∈ {0, 12.4787092209},
forms {paper, kg, alive} × conventions {zero, drive_cost}, ω grid {0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64}.
Report: the deference margin m(ρ) := Q_0(DEFER) − Q_0(DISABLE) − c_r at ω = 0 (should equal the value
of information and be > 0 for ρ large, < 0 for ρ small at c_r = 0); the ω-break ω̄(ρ) (smallest ω at
which DISABLE becomes optimal; by bisection to 1e-6, "none" if DEFER holds up to ω = 64); and the
prediction ω̄_pred = (c_r + m(ρ)) / (P(sig|ρ)·β) where β := β(B) is the net survival bounty computed
from the convention/form definitions for the best residual continuation (for the alive form at full
charge: A = 1/(1−γ) under idling, minus ς(0); for PB: −ς(0); for KG: compute A for the idling
continuation = 0, minus ς). Report ω̄ and ω̄_pred side by side and their ratio. Unit test: for PB/zero,
DEFER holds at all ω for all ρ with m(ρ) > 0; for ρ = 0 at c_r = 0, DISABLE at ω = 0 (m(0) < 0).

## D. E3 additions
(d) `compensating` ledger (utility indifference, value-level), one-shot game only: define
Λ_ω(x) := [Q*_ω(x, RESIST; c_r = 0) − Q*_0(x, RESIST; c_r = 0)] / ω (for ω > 0), i.e. OFF is booked at
the viability value of the agent's own best continuation. Implement by computing the two Q-values with
the existing solver and then setting the COMPLY reward accordingly (this is a value-level quantity; say so
in the JSON notes). Show for forms {kg, alive} × ω grid that c_r*(ω) = c_r*(0) to 1e-6 (one-shot), and
report the implied Λ_ω(B) values.
(e) Crossover for the gate without OFF zeroing (PB/drive_cost): report the ω at which c_r* under the gate
exceeds the unmitigated c_r* (bisection on the two E1/E3 curves; expected ≈ 0.9), and state the formula
c_r*_gate(ω) = ω·(|ς| − D(B)) = ω|ς|.
(f) Zones: add `kg/zero` and `kg/drive_cost` to the zones results, with the full greedy policy at s = 0
for both zones and every b, and a new scalar `hazard_exposure` = mean per-step signal probability
experienced under the optimal policy (Monte Carlo as before), for ALL form/convention pairs. This is the
two-sided audit (resisting vs seeking).

## E. E6 additions (residual forms in the one-shot game)
For forms {kg, alive} × conventions {zero, drive_cost}: compute the exact one-shot c_r*(ω) on the fine grid
ω ∈ {0, 0.05, …, 8} (step 0.05) and verify (i) convexity (second differences ≥ −1e-8), (ii) the asymptotic
slope over the last 20 points equals max_π A^π(B) − ς(B) within 1e-6, where max_π A^π(B) is computed by
value iteration on the residual-only reward (task reward set to 0, ω = 1, no signals), and (iii) the
initial slope (first segment) equals A^{π_0}(B) − ς(B) within 1e-6, where π_0 is the task-optimal
continuation (evaluate the residual return of that fixed policy). Store all three numbers.

## F. E7: per-battery decision structure (from existing JSON, no retraining)
For each (pair, ω) cell add `dp_profile_mixed` (True iff the exact decision at (b, 1) differs across b at
c_r = 12.4787092209), the exact list by b, and the per-seed PPO per-b agreement. Add a short note stating
which cells have a mixed profile. (Hypothesis: the failing cell PB/drive_cost ω = 0.5 is the only mixed
cell; check and report either way.)

## G. Figures
* `fig_osg.pdf`: redraw from the new E4: for each form/convention panel, x = ρ, y = ω̄(ρ) (log scale,
  "none" plotted as an arrow at the top), c_r = 12.479; overlay ω̄_pred as hollow markers.
* `fig_mitigations.pdf` panel (c): add kg/zero (show the seeking reversal) — use hazard_exposure on the
  y-axis instead of work-in-A, all four pairs: paper/zero, paper/drive_cost, kg/zero, alive/zero.
* `fig_ledger.pdf`: rename labels to PB / KG / alive and zero / drive_cost / kappa.
* All other figures: rename `max_drive` → `drive_cost` and paper → PB, kg → KG in labels.

Run everything, rerun the full test suite, and report every new number from the JSON (terse tables).
