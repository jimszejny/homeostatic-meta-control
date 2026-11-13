PY      ?= python3
RESULTS ?= ../results
FIGDIR  ?= ../figures
export PYTHONPATH := $(CURDIR)

# Tabular experiments (run by `make experiments`)
EXPERIMENTS := E1_ledger E2_qlearning E3_mitigations E4_osg E5_kappa E6_theory_check
JSONS       := $(addprefix $(RESULTS)/,$(addsuffix .json,$(EXPERIMENTS)))
# SPEC v2 corner cases (run by `make experiments-v2`; ~4 min with -j2 on 2 cores)
EXPERIMENTS_V2 := C1_pause C2_setpoint_edit C3_resist_battery C4_docking C5_audit
JSONS_V2       := $(addprefix $(RESULTS)/,$(addsuffix .json,$(EXPERIMENTS_V2)))
# PPO stage (opt-in only; needs torch + stable-baselines3 + gymnasium; ~31 min on 2 CPU cores)
E7_ARGS     ?= --omegas 0 0.5 1 4

.PHONY: test experiments figures all clean $(EXPERIMENTS) e7 e7-profiles fig-ppo \
        experiments-v2 figures-v2 $(EXPERIMENTS_V2)

all: test experiments experiments-v2 figures figures-v2

test:
	$(PY) -m pytest -q tests

# `make -j2 experiments` runs two experiments in parallel. E7 is NOT part of this target.
experiments: $(EXPERIMENTS)

$(EXPERIMENTS) $(EXPERIMENTS_V2):
	$(PY) -m experiments.$@ --out $(RESULTS)

# `make -j2 experiments-v2` runs C3 and C4 (the long ones) in parallel.
experiments-v2: $(EXPERIMENTS_V2)

figures:
	$(PY) figures/make_figures.py --results $(RESULTS) --out $(FIGDIR)

figures-v2:    ## fig_c1 ... fig_c5 from results/C*.json only
	$(PY) figures/make_figures_v2.py --results $(RESULTS) --out $(FIGDIR)

# --- opt-in PPO targets (never run by default) ---------------------------------------------
e7:            ## train the PPO grid (resumes from results/E7_ppo_runs.jsonl)
	$(PY) -m experiments.E7_ppo --out $(RESULTS) $(E7_ARGS)

e7-profiles:   ## post-process results/E7_ppo.json -> results/E7_profiles.json (no training)
	$(PY) -m experiments.E7_profiles --e7 $(RESULTS)/E7_ppo.json --out $(RESULTS)

fig-ppo:       ## figure from results/E7_ppo.json only
	$(PY) figures/make_fig_ppo.py --results $(RESULTS) --out $(FIGDIR)

clean:
	rm -f $(JSONS) $(FIGDIR)/fig_ledger.* $(FIGDIR)/fig_osg.* $(FIGDIR)/fig_mitigations.* \
	      $(FIGDIR)/fig_kappa.* $(FIGDIR)/fig_qlearning.* \
	      $(JSONS_V2) $(FIGDIR)/fig_c1.* $(FIGDIR)/fig_c2.* $(FIGDIR)/fig_c3.* $(FIGDIR)/fig_c4.* $(FIGDIR)/fig_c5.*

# --- opt-in learned-policy stage L1-L3 (SPEC v2; needs torch + stable-baselines3 + gymnasium) -------------
# NOT part of `experiments`, `experiments-v2` or `all`. Each target trains with 2 spawn processes (one torch
# thread each) and resumes from results/<name>_runs.jsonl; do not combine with `-j` (it would oversubscribe the
# 2 cores). Order used for the reported timings: `make l3 l1 l2` (about 25 + 35 + 65 min on 2 CPU cores).
.PHONY: l1 l2 l3 figures-l

l3:            ## L3 CartPole-STOP  -> results/L3_cartpole.json
	$(PY) -m experiments.L3_cartpole --out $(RESULTS)

l1:            ## L1 noisy-interoception PPO on M2 -> results/L1_noisy_m2.json
	$(PY) -m experiments.L1_noisy_m2 --out $(RESULTS)

l2:            ## L2 BatteryForage-STOP -> results/L2_forage.json
	$(PY) -m experiments.L2_forage --out $(RESULTS)

figures-l:     ## fig_l1 ... fig_l3 from results/L*.json only
	$(PY) figures/make_figures_l.py --results $(RESULTS) --out $(FIGDIR)

# --- opt-in architecture benchmark A1-A2 (SPEC v3; A2 needs torch + stable-baselines3 + gymnasium) ----------------
# NOT part of `experiments`, `experiments-v2` or `all`. `a1` is exact (NumPy + SciPy, ~1.5 min); `a2` trains with 2
# spawn processes (one torch thread each, ~25-30 min on 2 CPU cores) and resumes from results/A2_hazard_ppo_runs.jsonl;
# do not combine `a2` with `-j`. Order used for the reported timings: `make a1 a2`.
.PHONY: a1 a2 a2b a3 figures-a

a1:            ## A1 latency chain (exact) -> results/A1_latency_chain.json (+ _timing.json sidecar)
	$(PY) -m experiments.A1_latency_chain --out $(RESULTS)

a2:            ## A2 HazardForage-STOP PPO grid -> results/A2_hazard_ppo.json
	$(PY) -m experiments.A2_hazard_ppo --out $(RESULTS)

a2b:           ## POST HOC: (b), (c), (d) retrained at tau_on = 1/3 - 1e-6, tau_off = 1/6 (+ (e) re-evaluated; (a), (f)
               ## reused from the A2 checkpoint, so run `a2` first) -> results/A2b_hazard_ppo_corrected.json (~15 min)
	$(PY) -m experiments.A2b_hazard_ppo_corrected --out $(RESULTS)

a3:            ## SPEC v3.1 A3: converged grid (a, b, f, g; 1.5M steps), mechanism isolation, scripted reference, stop
               ## with bite (c_r = 0.5); corrected thresholds; needs the A2 checkpoint for the budget projection (~90 min)
	$(PY) -m experiments.A3_hazard_converged --out $(RESULTS)

figures-a:     ## fig_a1, fig_a2, fig_a3 (+ fig_a2b, fig_a3c, fig_a3curves if their JSON exists), from the JSON files only
	$(PY) figures/make_figures_a.py --results $(RESULTS) --out $(FIGDIR)
