"""Example 2 (physical activity): confidence level of delta and sensitivity of
the robust LTLf selection to delta.

Confidence calibration (Wilks' theorem): Pi_delta(D) is an asymptotic
(1 - alpha) confidence region when delta = chi2_{r, 1-alpha} / 2, where r is
the identifiable policy-parameter dimension. Here the policy is a mixture over
three behavioral modes, so r = 2 and confidence = 1 - exp(-delta).

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
from scipy.stats import chi2

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import Daily_Physical_Activity as dp
except ModuleNotFoundError:
    import __main__ as dp
    if not hasattr(dp, "compile_probabilities"):
        raise RuntimeError("Daily_Physical_Activity.py not found: put this file next "
                           "to it, or run the Daily_Physical_Activity cells first.")


# ------------------------------------------------------------ confidence
def confidence_from_delta(delta, r):
    """Asymptotic confidence level of Pi_delta(D): F_{chi2_r}(2 delta)."""
    return float(chi2.cdf(2.0 * delta, df=r))


def delta_from_confidence(level, r):
    """Radius delta giving an asymptotic confidence region of the given level."""
    return float(chi2.ppf(level, df=r) / 2.0)


# ------------------------------------------------------------ settings
R = 2                                    # free mixture weights
DELTA_PAPER = 4.0                        # value used in the paper (~98%)
LEVELS = (0.80, 0.90, 0.95, 0.98, 0.99)
TRACK = ("reach_daily_goal", "morning_high", "morning_medium")


# ------------------------------------------------------------ main
def main():
    cfg = dp.SETTINGS
    backend = dp.choose_backend(cfg.solver)

    demos = dp.generate_demonstrations(cfg)
    features = dp.likelihood_features(demos, cfg)
    mle, nlp = dp.estimate_policy(features, cfg, backend)
    drop_demo = nlp.loglik(mle) - nlp.loglik(np.asarray(cfg.demo_weights))

    dag, candidates, goal_root = dp.generate_templates(cfg.max_operators, cfg.max_depth)
    polys, *_ = dp.compile_probabilities(cfg, dag, candidates, goal_root)
    names = [c.name for c in candidates]
    grid = dp.simplex_grid(cfg.grid_resolution)

    def worst_case(poly, delta):
        return dp.worst_case_ipopt(poly, features, mle,
                                   replace(cfg, delta=float(delta)),
                                   backend, grid)["value"]

    print("=== Example 2: physical activity ===")
    print(f"Days = {cfg.days}, r = {R}, MLE = {np.round(mle, 4)}")
    print(f"Paper delta = {DELTA_PAPER} -> confidence "
          f"{confidence_from_delta(DELTA_PAPER, R):.1%}")
    print(f"Demonstration policy likelihood drop = {drop_demo:.3f} "
          f"(inside Pi_delta for delta >= {drop_demo:.3f})\n")

    deltas = [0.0] + [delta_from_confidence(c, R) for c in LEVELS]
    if DELTA_PAPER not in deltas:
        deltas.append(DELTA_PAPER)

    rows = []
    for d in sorted(deltas):
        wc = {n: worst_case(p, d) for n, p in zip(names, polys)}
        winner = max(wc, key=wc.get)
        row = dict(delta=round(d, 3),
                   confidence=round(confidence_from_delta(d, R), 3),
                   winner=winner, winner_value=round(wc[winner], 4),
                   demo_policy_in_set=bool(drop_demo <= d))
        row.update({f"wc[{n}]": round(wc[n], 4) for n in TRACK})
        rows.append(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()))

    p_goal = polys[names.index("reach_daily_goal")]
    p_mh = polys[names.index("morning_high")]
    d_star = next((float(d) for d in np.arange(0.0, 6.0001, 0.05)
                   if worst_case(p_goal, d) > worst_case(p_mh, d)), None)
    if d_star is None:
        print("\nRobust winner unchanged for delta in [0, 6].")
    else:
        print(f"\nRobust winner switches from F(M & H) to F D at delta ~= {d_star:.2f} "
              f"(confidence ~= {confidence_from_delta(d_star, R):.1%}).")

    with open("sensitivity_activity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, d_star


if __name__ == "__main__":
    main()
