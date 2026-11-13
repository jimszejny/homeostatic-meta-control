# Pre-specified predictions -- sigma module (regime chain)

Written: 2026-11-11T17:28:45Z (UTC), after Step 0 passed (results/step0.json) and before any E1-E5 run.
This file is not edited after hashing; corrections go into REPORT.md.

## Brief's predictions (verbatim)

Q1. For every lambda < 1 and theta > lambda, the filtered-gate reflex has strictly more hits than the raw-gate reflex; for theta <= lambda it has the same hits.
Q2. At baseline (G=0.8, r_g=0.7, c_sw=0) the best sigma controller beats both fixed modes in reward per step.
Q3. The best lambda is interior: strictly between 1 and the smallest value tested, and both lambda=1 and lambda=0.005 capture less than half of the gain the best lambda captures.
Q4. The best sigma controller captures at least 70% of the Bayes controller's gain over the best fixed mode at c_sw=0.
Q5. With a perfect reflex (G=1) at r_g=0.7 no mode-switching controller, Bayes included, beats always-WORK.
Q6. Without a fuse, the long-run GUARD fraction exceeds 0.9 for rho >= 0.2; with the fuse (T_max=100, P=20) it stays below 0.5 for every rho tested, and the fuse costs less than 10% of the sigma controller's gain at rho=0.
Q7. c*_r(b) is decreasing in the storm belief b for both controllers, and the Bayes controller's stationary-average c*_r is higher than always-WORK's.

## Fixed evaluation choices (declared before running)

- "sigma controller" in Q2-Q4 = the literal timing of the brief: the mode committed at t uses sigma_t, which contains a_0..a_{t-1} (one announcement less than the Bayes controller). Best over lambda and over single- and two-threshold latches. An information-matched variant (mode at t uses sigma_{t+1}, i.e. a_0..a_t) is reported as a secondary row and is NOT used to score Q2-Q4.
- Gains are long-run average reward per step; "best fixed mode" = max(always-WORK, always-GUARD).
- E4 uses the best (lambda, theta_on, theta_off) of E2 at c_sw=0 (two-threshold search, which contains single thresholds), unchanged for every rho.
- Oracle = knows the regime of the announcement that governs the impact at the execution step.

## My own predictions (marked as mine)

M1 (mine). The best lambda lies in [0.05, 0.2] at baseline.
M2 (mine). The information-matched sigma captures a strictly larger fraction of the Bayes gain than the literal sigma at c_sw=0.
M3 (mine). At c_sw=1 the Bayes belief-MDP policy is two-threshold (b_off < b_on), and the best two-threshold sigma beats the best single-threshold sigma by more than its Monte-Carlo SE.
M4 (mine). Without the fuse, the GUARD fraction rises steeply in rho near rho = theta_off of the E4 configuration (mean-field lock-in condition), not near theta_on.
M5 (mine). In E5 the indifference shortcut is not valid at most beliefs: the true c*_r(b) from the full stopping problem is >= the shortcut value, with equality only near the lowest (calmest) beliefs.
