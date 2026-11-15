"""Example 2 (physical activity): sensitivity of the robust selection to delta.

The demonstrations, model, and candidate set are kept fixed while delta is
increased from 0 to 4 in steps of 0.25. For delta = 0 the policy set contains
only the maximum-likelihood mixture, so the evaluation is the nominal one. For
each delta, the script reports the selected formula and the five
highest-ranked formulas, and it prints the range of tested delta values over
which the selected formula stays the same as at the largest delta.

Works either as a script next to Daily_Physical_Activity.py, or in a notebook
after running the Daily_Physical_Activity cells (the functions are then taken
from the notebook namespace).

Output: printed table and sensitivity_activity.csv
"""
import contextlib
import csv
import io
from dataclasses import replace

import numpy as np

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import Daily_Physical_Activity as dp
except ModuleNotFoundError:
    import __main__ as dp
    if not hasattr(dp, "compile_probabilities"):
        raise RuntimeError("Daily_Physical_Activity.py not found: put this file next "
                           "to it, or run the Daily_Physical_Activity cells first.")

DELTAS = np.round(np.arange(0.0, 4.0 + 1e-9, 0.25), 2)
TOP = 5


def stable_range(deltas, winners):
    """Smallest tested delta from which the selection equals the final one."""
    k = len(winners) - 1
    while k > 0 and winners[k - 1] == winners[-1]:
        k -= 1
    return deltas[k], deltas[-1], winners[-1]


def main(deltas=DELTAS):
    cfg = dp.SETTINGS
    backend = dp.choose_backend(cfg.solver)

    demos = dp.generate_demonstrations(cfg)
    features = dp.likelihood_features(demos, cfg)
    mle, _ = dp.estimate_policy(features, cfg, backend)

    dag, candidates, goal_root = dp.generate_templates(cfg.max_operators, cfg.max_depth)
    polys, *_ = dp.compile_probabilities(cfg, dag, candidates, goal_root)
    names = [c.name for c in candidates]
    grid = dp.simplex_grid(cfg.grid_resolution)

    def worst_case(poly, delta):
        return dp.worst_case_ipopt(poly, features, mle,
                                   replace(cfg, delta=float(delta)),
                                   backend, grid)["value"]

    rows, winners = [], []
    for d in deltas:
        vals = {n: worst_case(p, d) for n, p in zip(names, polys)}
        ranking = sorted(vals, key=vals.get, reverse=True)
        winners.append(ranking[0])
        rows.append({"delta": float(d), "selected": ranking[0],
                     "top5": " > ".join(ranking[:TOP]),
                     **{n: round(vals[n], 4) for n in names}})
        print(f"delta = {d:4.2f}  selected: {ranking[0]:18s} "
              f"({vals[ranking[0]]:.3f})  top {TOP}: "
              + ", ".join(f"{n} {vals[n]:.3f}" for n in ranking[:TOP]),
              flush=True)

    lo, hi, w = stable_range(list(deltas), winners)
    print(f"\nThe selected formula is {w} for all tested values of delta "
          f"from {lo:g} to {hi:g}.")

    with open("sensitivity_activity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
