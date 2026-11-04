"""Example 1 (route selection): confidence level of delta and sensitivity of
the robust LTLf selection to delta.

Confidence calibration (Wilks' theorem): Pi_delta(D) is an asymptotic
(1 - alpha) confidence region when delta = chi2_{r, 1-alpha} / 2, where r is
the identifiable policy-parameter dimension. Here the policy is a distribution
over three route families, so r = 2 and confidence = 1 - exp(-delta).

Works either as a script next to Route_Selection.py, or in a notebook after
running the Route_Selection cells (the functions are then taken from the
notebook namespace).

Output: printed table and sensitivity_route.csv
"""
import contextlib
import csv
import io

import numpy as np
from scipy.stats import chi2

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import Route_Selection as rs
except ModuleNotFoundError:
    import __main__ as rs
    if not hasattr(rs, "worst_case_ipopt"):
        raise RuntimeError("Route_Selection.py not found: put this file next to it, "
                           "or run the Route_Selection notebook cells first.")


# ------------------------------------------------------------ confidence
def confidence_from_delta(delta, r):
    """Asymptotic confidence level of Pi_delta(D): F_{chi2_r}(2 delta)."""
    return float(chi2.cdf(2.0 * delta, df=r))


def delta_from_confidence(level, r):
    """Radius delta giving an asymptotic confidence region of the given level."""
    return float(chi2.ppf(level, df=r) / 2.0)


# ------------------------------------------------------------ settings
R = 2                                    # free parameters of the route distribution
DELTA_PAPER = 3.0                        # value used in the paper (~95%)
LEVELS = (0.80, 0.90, 0.95, 0.98, 0.99)  # confidence levels for the sweep
N_TRAJS, P_TRUE, SEED = 80, (0.72, 0.14, 0.14), 18
TRACK = ("F T", "F C", "F S")


# ------------------------------------------------------------ helpers
def worst_case(counts, phi, delta, restarts=25, seed=0):
    values = rs.ROUTE_FORMULA_VALUES[phi]
    if delta == 0:                       # delta = 0 -> nominal (MLE) value
        return float(np.dot(values, rs.mle_from_counts(counts)))
    return rs.worst_case_ipopt(counts=counts, values=values, delta=delta,
                               restarts=restarts, seed=seed, max_iter=400)["val_wc"]


def robust_selection(counts, delta):
    wc = {phi: worst_case(counts, phi, delta, seed=10 + i)
          for i, phi in enumerate(rs.FORMULAS)}
    return max(wc, key=wc.get), wc


def crossover(counts, a="F T", b="F C", grid=np.arange(0.0, 6.0001, 0.05)):
    """Smallest delta on the grid at which formula a overtakes formula b."""
    for d in grid:
        if worst_case(counts, a, d, restarts=8) > worst_case(counts, b, d, restarts=8):
            return float(d)
    return None


# ------------------------------------------------------------ main
def main():
    counts = rs.generate_first_action_counts(N_TRAJS, P_TRUE, SEED)
    ll_hat = rs.multinomial_loglik(rs.mle_from_counts(counts), counts)
    drop_true = ll_hat - rs.multinomial_loglik(np.array(P_TRUE), counts)

    print("=== Example 1: route selection ===")
    print(f"Demonstration counts = {counts}, r = {R}")
    print(f"Paper delta = {DELTA_PAPER} -> confidence "
          f"{confidence_from_delta(DELTA_PAPER, R):.1%}")
    print(f"Generating policy likelihood drop = {drop_true:.3f} "
          f"(inside Pi_delta for delta >= {drop_true:.3f})\n")

    deltas = [0.0] + [delta_from_confidence(c, R) for c in LEVELS]
    if DELTA_PAPER not in deltas:
        deltas.append(DELTA_PAPER)

    rows = []
    for d in sorted(deltas):
        winner, wc = robust_selection(counts, d)
        row = dict(delta=round(d, 3),
                   confidence=round(confidence_from_delta(d, R), 3),
                   winner=winner, winner_value=round(wc[winner], 4),
                   generating_policy_in_set=bool(drop_true <= d))
        row.update({f"wc[{phi}]": round(wc[phi], 4) for phi in TRACK})
        rows.append(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()))

    d_star = crossover(counts)
    if d_star is None:
        print("\nRobust winner unchanged for delta in [0, 6].")
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
