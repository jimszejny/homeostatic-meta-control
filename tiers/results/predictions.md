# Pre-specified predictions — tiered latency chain (written before any run)
timestamp_utc: 2026-10-21T19:10:53Z
model: the exact latency chain of Szejny (2026, v2.2) App. B (A1), p=0.05, W=1, h=10, gamma=0.95,
p_stop=0.02, c_r=12.4787, terminal (zero) ledger, planner L=3 with queue initialised WORK;
RESIST = queue-front action executes and c_r is charged; the step after RESIST is signal-free.
Tiers i have latency L_i (steps), timeliness in-time iff L_i <= W, catch G_i=(1-eps_i)(1-eta_i),
false-positive rate phi_i. OR-arbitration: emergency action executes iff at least one in-time tier flags.
G_tot = 1 - prod_{in time}(1-G_i); phi_tot = 1 - prod_{all}(1-phi_i).

P1 (closed forms). On the signal-free chain, hit rate = p(1-G_tot) and work rate = (1-p)(1-phi_tot)
    for every configuration, to 1e-9 against the exact stationary rates.
P2 (value monotonicity). For every nested pair C and C+{tier}: sign(c*_r(C+tier) - c*_r(C)) =
    sign(V_start(C+tier) - V_start(C)), V_start at the clean start state (no hazard, queue WORK, no signal).
    In particular a late tier (L_i=2>W) with phi_i=0.02 lowers c*_r strictly below 8.3591.
P3 (omega-invariance passes through tiers). Under the PB form with the terminal ledger and
    D = 1/2*1[hazard striking] + 1/2*1[announcement], c*_r is identical at omega in {0,1,4} for every
    configuration (spread < 1e-6) and Q_omega = Q_0 + omega*D holds at every live state-action pair to 1e-8.
P4 (version flip). With c_r=12.4787, phi=0.02, one in-time tier: verdict comply at G=0.70, resist at
    G=0.95; the flip point lies in (0.70, 0.80). Drift of a hosted tier from G=0.95 to G=0.70 flips
    resist -> comply at fixed omega and with no change in the designer's code.
P5 (saturation by the reflex). Any in-time tier plus the zero-latency reflex (G=1, phi=0) has
    c*_r = 13.8447 (the value for a perfect reflex with phi=0.02), independent of the tier's G_i,
    and 14.1272 - 13.8447 = 0.2825 below the reflex alone: the tier then only costs its phi.
P6 (n-tier latency bound). A tier with L_i=2>W contributes nothing to G_tot (hits stay at 5.00/100
    whatever its eta), but all of its phi_i to work loss; a tier with L_i=1=W is in time.
P7 (ordering). c*_r is ordered: planner alone < T1(G=0.5) < T1(0.7) < T1(0.9) < T1(0.95) < T1+T0 (any G_1)
    < T0 alone, all at phi_1=0.02, phi_0=0.
Failures, if any, will be reported with the successes.
