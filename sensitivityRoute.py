"""Example 1 (route navigation): sensitivity of the robust selection to delta.

The demonstrations, MDP, and candidate set are kept fixed while delta is
increased from 0 to 4 in steps of 0.25. For delta = 0 the policy set contains
only the maximum-likelihood policy, so the evaluation is the nominal one. For
each delta, the script reports the selected formula and the five
highest-ranked formulas, and it prints the range of tested delta values over
which the selected formula stays the same as at the largest delta.

Put this file next to RouteSelection.py.
Output: printed table and sensitivity_route.csv
"""
import csv

import numpy as np

import RouteSelection as rs

DELTAS = np.round(np.arange(0.0, 4.0 + 1e-9, 0.25), 2)
N_TRAJS, SEED = 80, 18          # same demonstrations as in the paper
TOP = 5


def robust_values(model, counts, theta_hat, delta):
    """Worst-case satisfaction probability of every candidate formula."""
    if delta == 0:
        return model.all_values(theta_hat)
    return {phi: rs.worst_case_multistart(phi, counts, theta_hat, delta, model,
                                          seed=10 + i)["val_wc"]
            for i, phi in enumerate(rs.FORMULAS)}


def stable_range(deltas, winners):
    """Smallest tested delta from which the selection equals the final one."""
    k = len(winners) - 1
    while k > 0 and winners[k - 1] == winners[-1]:
        k -= 1
    return deltas[k], deltas[-1], winners[-1]


def main(deltas=DELTAS):
    model = rs.SatisfactionModel()
    counts = rs.generate_demonstrations(N_TRAJS, rs.THETA_TRUE, SEED)
    theta_hat = rs.mle(counts)

    rows, winners = [], []
    for d in deltas:
        vals = robust_values(model, counts, theta_hat, float(d))
        ranking = sorted(vals, key=vals.get, reverse=True)
        winners.append(ranking[0])
        rows.append({"delta": float(d), "selected": ranking[0],
                     "top5": " > ".join(ranking[:TOP]),
                     **{phi: round(vals[phi], 4) for phi in rs.FORMULAS}})
        print(f"delta = {d:4.2f}  selected: {ranking[0]:12s} "
              f"({vals[ranking[0]]:.3f})  top {TOP}: "
              + ", ".join(f"{phi} {vals[phi]:.3f}" for phi in ranking[:TOP]),
              flush=True)

    lo, hi, w = stable_range(list(deltas), winners)
    print(f"\nThe selected formula is {w} for all tested values of delta "
          f"from {lo:g} to {hi:g}.")

    with open("sensitivity_route.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
