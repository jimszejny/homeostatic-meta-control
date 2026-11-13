"""offledger: shutdown incentives in homeostatic RL agents (SPEC v1.0 + v1.1 addendum + v1.2 round 2
+ v2.0 corner cases C1-C5).

The tabular part (m1, m2, solve, qlearn) depends only on NumPy. ``offledger.gym_env``
imports Gymnasium lazily and is never imported from here.
"""
SPEC_VERSION = "2.0"   # package version of the spec
# spec_version written into each results JSON = the spec revision that last changed that file's
# content (so that re-running an unchanged experiment stays byte-identical).
RESULT_SPEC_VERSION = {"E3_mitigations": "1.2", "E4_osg": "1.2",
                       "C1_pause": "2.0", "C2_setpoint_edit": "2.0", "C3_resist_battery": "2.0",
                       "C4_docking": "2.0", "C5_audit": "2.0"}
RESULT_SPEC_VERSION_DEFAULT = "1.1"
