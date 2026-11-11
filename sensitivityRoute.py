"""Example 1 (route navigation): confidence level of delta and sensitivity of
the robust LTLf selection to delta.

Confidence calibration (Wilks' theorem): Pi_delta(D) is an asymptotic
(1 - alpha) confidence region when delta = chi2_{r, 1-alpha} / 2, where r is
the number of identifiable policy parameters. Here the policy is the softmax
policy of Eq. (1) with d = 4 features, so r = 4 and
confidence = 1 - exp(-delta) * (1 + delta).

Put this file next to RouteSelection.py (the Example 1 code).

Output: printed table and sensitivity_route.csv
"""
import csv

import numpy as np
from scipy.stats import chi2

import Route_Selection as rs


# ------------------------------------------------------------ confidence
def confidence_from_delta(delta, r):
    """Asymptotic confidence level of Pi_delta(D): F_{chi2_r}(2 delta)."""
    return float(chi2.cdf(2.0 * delta, df=r))


def delta_from_confidence(level, r):
    """Radius delta giving an asymptotic confidence region of the given level."""
    return float(chi2.ppf(level, df=r) / 2.0)


# ------------------------------------------------------------ settings
R = rs.D                                 # number of policy parameters (4)
DELTA_PAPER = 4.0                        # value used in the paper (~91%)
LEVELS = (0.80, 0.90, 0.95, 0.98, 0.99)  # confidence levels for the sweep
N_TRAJS, SEED = 80, 18
TRACK = ("F T", "F C", "F S")


# ------------------------------------------------------------ helpers
def worst_case(model, counts, theta_hat, phi, delta, seed=0):
    if delta == 0:                       # delta = 0 -> nominal (MLE) value
        return model.value(theta_hat, rs.ACCEPT[phi])
    return rs.worst_case_multistart(phi, counts, theta_hat, delta, model,
                                    seed=seed)["val_wc"]


def robust_selection(model, counts, theta_hat, delta):
    wc = {phi: worst_case(model, counts, theta_hat, phi, delta, seed=10 + i)
          for i, phi in enumerate(rs.FORMULAS)}
    return max(wc, key=wc.get), wc


def crossover(model, counts, theta_hat, a="F T", b="F C", hi=2.0, tol=0.01):
    """Smallest delta at which formula a overtakes formula b (bisection)."""
    def gap(d):
        return (worst_case(model, counts, theta_hat, a, d)
                - worst_case(model, counts, theta_hat, b, d))
    if gap(0.0) > 0:
        return 0.0
    if gap(hi) <= 0:
        return None
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------ main
def main():
    model = rs.SatisfactionModel()
    counts = rs.generate_demonstrations(N_TRAJS, rs.THETA_TRUE, SEED)
    theta_hat = rs.mle(counts)
    drop_true = rs.loglik(theta_hat, counts) - rs.loglik(rs.THETA_TRUE, counts)

    print("=== Example 1: route navigation ===")
    print(f"Demonstrations = {N_TRAJS}, r = {R}")
    print(f"Paper delta = {DELTA_PAPER} -> confidence "
          f"{confidence_from_delta(DELTA_PAPER, R):.1%}")
    print(f"Generating policy likelihood drop = {drop_true:.3f} "
          f"(inside Pi_delta for delta >= {drop_true:.3f})\n")

    deltas = [0.0] + [delta_from_confidence(c, R) for c in LEVELS]
    if DELTA_PAPER not in deltas:
        deltas.append(DELTA_PAPER)

    rows = []
    for d in sorted(deltas):
        winner, wc = robust_selection(model, counts, theta_hat, d)
        row = dict(delta=round(d, 3),
                   confidence=round(confidence_from_delta(d, R), 3),
                   winner=winner, winner_value=round(wc[winner], 4),
                   generating_policy_in_set=bool(drop_true <= d))
        row.update({f"wc[{phi}]": round(wc[phi], 4) for phi in TRACK})
        rows.append(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    d_star = crossover(model, counts, theta_hat)
    if d_star is None:
        print("\nF T does not overtake F C for delta in [0, 2].")
    else:
        print(f"\nRobust winner switches from F C to F T at delta ~= {d_star:.2f} "
              f"(confidence ~= {confidence_from_delta(d_star, R):.1%}).")

    with open("sensitivity_route.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, d_star


if __name__ == "__main__":
    main()
