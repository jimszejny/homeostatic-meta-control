# Pre-specified predictions -- REWARD module (off-state ledger)

Written (UTC): 2026-11-11T17:31:13Z
Status at writing: Step 0 has passed (planner alone V_start 8.2119, c*_r 8.3591; planner + reflex
14.4440, 14.1272; K 12.8181, A 14.1272, B 5.7682; omega-spread 0 / 1.1e-14). No part of R1-R4 has been
run. This file is not edited after its hash is recorded in predictions.sha256.

## Specified predictions (verbatim from the task)

R-P1. Under L0 and L0p, c*_r is identical for all omega (spread < 1e-9); under L1, L2, L3 it increases linearly in omega.

R-P2. On the battery chain the potential-based term with death at zero potential leaves the optimal policy identical for every omega; the residual forms raise mean energy monotonically in their weight.

R-P3. With delta > 0, death priced (D=1) and off neutral, c*_r decreases in omega whenever Gamma < 1, and is independent of omega when Gamma = 1.

R-P4. The closed forms derived for beta and for the death-state extension match bisection to 1e-9 wherever admissibility holds.

## Own predictions (marked as mine, written before the runs)

R-P5 (mine). Under L4 (residual drive rent) c*_r decreases linearly in omega with slope
beta_L4 = -(1-gamma) p (1/2 + gamma + K) = -0.0357 (both planner alone and planner + reflex).

R-P6 (mine). For every ledger L0-L4 on the base chain, beta does not depend on Gamma or phi_tot:
the planner-alone and planner+reflex slopes are equal to 1e-9. Predicted values:
beta_L0 = beta_L0p = 0, beta_L1 = gamma(1-M) = 0.7015, beta_L2 = b(1+gamma+K) = 1.4768,
beta_L3 = kappa(1-M) = 0.7384, with M = gamma^2 p_sig/(1-gamma(1-p_sig)) = 0.2616.

R-P7 (mine). Battery chain as suggested (B1: E=10, d=0.8, recharge +2 free, gamma=0.95) is
degenerate for kappa*: without any death penalty the optimal policy never WORKs at e=1, for every
gamma in {0.8,0.9,0.95,0.99}, so kappa* = 0. Committed adjustment if (and only if) kappa*(0.95)=0 on
B1: B2 = B1 with RECHARGE paying -c_R, c_R = 2 (a price of energy), nothing else changed; both B1
and B2 are reported.

R-P8 (mine). Death chain (R4), death booked at zero potential, off neutral: c*_r independent of
omega (spread < 1e-9) and, for Gamma < 1, strictly decreasing in delta.

R-P9 (mine). Death chain with off priced as in L1 and death priced at D=1: c*_r increases in omega
for every configuration in the grid (the off price dominates the death price).

## Procedure commitments
- c_r = 12.4787 wherever a verdict is needed; bisection on [0,300], 60 halvings; policy iteration
  with exact linear solves (value iteration cross-check).
- Omega grid {0, 0.5, 1, 2, 4}; any extended sweep is labelled as such.
- No parameter tuning after seeing results; any post-hoc analysis is labelled post hoc.
