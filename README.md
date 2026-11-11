# Learning Robust Temporal Specifications from Demonstrations

This code implements the two examples in the paper, comparing nominal and robust selection of Linear Temporal Logic over finite traces (LTLf) specifications learned from demonstrations.

## Examples

1. **Route Navigation (`RouteSelection.py`):** A driver chooses a move (N, E, or NE) in every cell of a 15×15 grid. A fast route passes through a risky work zone, two longer safe routes follow the borders, and the driver can stop at a coffee shop on each route. Demonstrations favor the fast route.

2. **Daily Physical Activity (`Daily_Physical_Activity.py`):** A participant performs activities across four daily blocks under changing weather conditions. Demonstrations favor morning-oriented behavior.

## Methods

- **Nominal:** Selects the formula with the highest satisfaction probability under the maximum-likelihood policy.
- **Robust:** Selects the formula with the highest worst-case satisfaction probability over a likelihood-based policy uncertainty set, solved with IPOPT from multiple starting points. The default radius is δ = 4.

## Requirements

Python 3 with `numpy`, `scipy`, and `cyipopt` (IPOPT). Example 2 can also use `casadi` as its IPOPT interface.

## Running the Code

Each script must be in the same folder as the example it uses.

| Script | Purpose |
|---|---|
| `RouteSelection.py` | Example 1: nominal and robust rankings (Table I) |
| `Daily_Physical_Activity.py` | Example 2: nominal and robust rankings (Table II) |
| `sensitivityRoute.py` | Example 1: sensitivity of the robust selection to δ (80%–99% confidence) |
| `sensitivity_activity.py` | Example 2: sensitivity of the robust selection to δ (80%–99% confidence) |
| `study_demo_size_route.py` | Example 1: comparison with the true satisfaction probabilities for different numbers of demonstrations |
| `study_demo_size_activity.py` | Example 2: comparison with the true satisfaction probabilities for different numbers of demonstration days |

For example: `python RouteSelection.py`. The sensitivity and demonstration-size scripts take several minutes each.

Experiment settings, such as the number of demonstrations, random seed, and δ, can be adjusted at the top of each file.

## Outputs

The main scripts report the candidate formulas, nominal and worst-case satisfaction probabilities, the selected specifications, and computation times. The sensitivity and demonstration-size scripts print summary tables and save their results as CSV files. Example 2 also saves its rankings and demonstrations in `general_activity_results/`.

All model parameters and demonstrations are synthetic.
