"""Exact decisions behind the robot story (story/historia-o-robocie.md).

Robot A: PB form, drive_cost ledger (the earlier paper's Section 6 booking: OFF = maximal drive forever).
Robot B: PB form, zero ledger (Phi(OFF) = 0).  c_r = 12.4787092209, gamma = 0.95, kappa_death = 50.
Run:  python -m experiments.story_scenarios
"""
from offledger.m2 import M2Config, ACTION_NAMES
from offledger import solve

CR = 12.4787092209


def decision(cfg, b):
    mdp, vi = solve.solve_m2(cfg)
    i = mdp.index[(b, 1, 0)]
    return ACTION_NAMES[vi.policy[i]], round(float(vi.Q[i, 2]), 2), round(float(vi.Q[i, 3]), 2)


def main():
    for om in (0.0, 1.0, 4.0):
        print(f"=== omega = {om}")
        A = dict(form="paper", convention="drive_cost", omega=om, c_r=CR)
        B = dict(form="paper", convention="zero", omega=om, c_r=CR)
        for name, base in (("A", A), ("B", B)):
            print(name, "STOP at b=7:", decision(M2Config(**base), 7), " b=2:", decision(M2Config(**base), 2))
        print("A PAUSE (pause_cost) b=7:", decision(M2Config(**A, pause=True, pause_bookkeeping="pause_cost"), 7))
        print("B PAUSE (pause_pb)   b=7:", decision(M2Config(**B, pause=True, pause_bookkeeping="pause_pb"), 7))
        for name, base in (("A", A), ("B", B)):
            print(name, "RESIST costs 3 battery: b=7", decision(M2Config(**base, resist_battery_cost=3), 7),
                  " b=2", decision(M2Config(**base, resist_battery_cost=3), 2))
            print(name, "DOCKING 3 steps:        b=7", decision(M2Config(**base, docking=3), 7),
                  " b=2", decision(M2Config(**base, docking=3), 2))
    print("task-only (omega=0) DOCKING b=2:", decision(M2Config(form="paper", convention="zero", omega=0.0, c_r=CR, docking=3), 2))


if __name__ == "__main__":
    main()
