# Learning Robust Temporal Specifications from Demonstrations

This code implements the two examples in the paper, comparing nominal and robust selection of Linear Temporal Logic over finite traces (LTLf) specifications learned from demonstrations. It also includes the sensitivity analysis for δ and a ground-truth experiment for Example 1.

## Examples

1. **Route Navigation (`RouteSelection.py`):** A driver chooses a move (N, E, or NE) in every cell of a 15×15 grid. A fast route passes through a risky work zone, two longer safe routes follow the borders, and the driver can stop at a coffee shop on each route. Demonstrations favor the fast route.

2. **Daily Physical Activity (`Daily_Physical_Activity.py`):** A participant performs activities across four daily blocks under changing weather conditions. Demonstrations favor morning-oriented behavior.

## Methods

- **Nominal:** Selects the formula with the highest satisfaction probability under the maximum-likelihood policy.
- **Robust:** Selects the formula with the highest worst-case satisfaction probability over a likelihood-based policy uncertainty set, solved with IPOPT from multiple starting points. The default radius is δ = 2.

## Requirements

Python 3 with `numpy`, `scipy`, and `cyipopt` (IPOPT). Example 2 can also use `casadi` as its IPOPT interface.

## Running the Code

Each script must be in the same folder as the example it uses.

| Script | Purpose |
|---|---|
| `RouteSelection.py` | Example 1: nominal and robust rankings (Table I) |
| `Daily_Physical_Activity.py` | Example 2: nominal and robust rankings (Table II) |
| `sensitivityRoute.py` | Example 1: robust selection for δ from 0 to 4 in steps of 0.25 (δ = 0 is the nominal evaluation) |
| `sensitivity_activity.py` | Example 2: robust selection for δ from 0 to 4 in steps of 0.25 (δ = 0 is the nominal evaluation) |
| `ground_truth_perturbation_EX1.py` | Example 1: ground-truth experiment with known intended formulas (see below) |
| `study_demo_size_route.py` | Example 1: comparison with the true satisfaction probabilities for different numbers of demonstrations |
| `study_demo_size_activity.py` | Example 2: comparison with the true satisfaction probabilities for different numbers of demonstration days |

For example: `python RouteSelection.py`. The sensitivity, ground-truth, and demonstration-size scripts take from several minutes to about an hour each.

Experiment settings, such as the number of demonstrations, random seed, and δ, can be adjusted in `run()` in `RouteSelection.py` and in `Settings` in `Daily_Physical_Activity.py`. The δ values tested by the sensitivity scripts are set by `DELTAS` at the top of each script.

## Ground-Truth Experiment (Example 1)

`ground_truth_perturbation_EX1.py` tests both methods when the intended formula is known in advance. It uses the model, likelihood, and solver of `RouteSelection.py`.

For each intended formula φ*:

1. **Expert policy:** the policy in the softmax policy class of Example 1 that maximizes the satisfaction probability of φ*, computed with the known transition probabilities. A small penalty on the policy parameters leaves behavior that φ* does not constrain random.
2. **Perturbation:** the expert policy is perturbed slightly toward random actions. The perturbed expert is the policy in the same class that behaves most like an expert who takes a uniformly random action with probability ε at every step.
3. **Demonstrations:** for each number of demonstrations N, independent sets of N demonstrations are drawn from the perturbed expert.
4. **Selection:** nominal and robust selection are applied to each set, using all 26 candidate formulas of Example 1.
5. **Comparison:** each selected formula is compared with φ* (equal to φ*, a weaker formula implied by φ*, or not implied), and the value each method reports is compared with the true satisfaction probability of the selected formula under the perturbed expert.

Settings at the top of the script:

| Setting | Value in the paper | Meaning |
|---|---|---|
| `GROUND_TRUTHS` | F T, F C, F S, F T ∧ G¬C, F C ∧ G¬T, F T ∧ F S | intended formulas |
| `EPSILONS` | 0.0005, 0.001 | perturbation levels |
| `NS` | 20, 50, 80 | numbers of demonstrations |
| `SEEDS` | 2 | independent demonstration sets per formula, ε, and N (4 sets per formula and N in total) |
| `DELTA` | 2 and 4 (one run each) | radius of the uncertainty set |
| `CANDIDATES` | `None` (all 26 formulas) | formulas both methods choose from |

The script runs in parallel on all CPU cores and can be resumed: finished demonstration sets are saved and skipped when it is rerun. Delete the output file after changing a setting. Commands:

- `python ground_truth_perturbation_EX1.py` runs the experiment and prints a summary.
- `python ground_truth_perturbation_EX1.py summary` prints the summary of existing results only.
- `python ground_truth_perturbation_EX1.py validate` checks the solver against `RouteSelection.py`.

The results reported in the paper are in `ground_truth_perturbation_route_delta2.csv` and `ground_truth_perturbation_route_delta4.csv`, one row per demonstration set. The main columns are:

| Column | Meaning |
|---|---|
| `truth`, `eps`, `N`, `seed` | intended formula, perturbation level, number of demonstrations, and set index |
| `nominal_pick`, `robust_pick` | formula selected by each method |
| `nominal_relation`, `robust_relation` | `= phi*`, `implied by phi*`, or `not implied` |
| `nominal_value`, `robust_value` | satisfaction probability reported by each method |
| `nominal_true`, `robust_true` | true satisfaction probability of the selected formula under the perturbed expert |
| `theta_star_in_set` | whether the unperturbed expert policy lies in the uncertainty set (log-likelihood within δ of the maximum) |

## Outputs

The main scripts report the candidate formulas, nominal and worst-case satisfaction probabilities, the selected specifications, and computation times. The sensitivity scripts print the selected formula and the five highest-ranked formulas for each δ, followed by the range of tested δ values over which the selected formula does not change. The ground-truth script prints the expert policy for each intended formula and a summary of both methods' selections. The sensitivity, ground-truth, and demonstration-size scripts save their results as CSV files. Example 2 also saves its rankings and demonstrations in `general_activity_results/`.

All model parameters and demonstrations are synthetic.
