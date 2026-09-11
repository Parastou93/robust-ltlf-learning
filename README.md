# Learning Robust Temporal Specifications from Demonstrations

This code implements two examples comparing nominal and robust selection of Linear Temporal Logic over finite traces (LTLf) specifications.

## Examples

1. **Route Selection:** A driver chooses among two safe routes and a shortcut through a work zone, with optional coffee-shop stops. Demonstrations favor the shortcut.

2. **Daily Physical Activity:** A participant performs activities across four daily blocks under changing weather conditions. Demonstrations favor morning-oriented behavior.

## Methods

- **Nominal:** Selects the formula with the highest satisfaction probability under the maximum-likelihood estimate.
- **Robust:** Selects the formula with the highest worst-case satisfaction probability over a likelihood-based policy uncertainty set, using IPOPT.

## Running the Code

Install the dependencies imported by the code, including its IPOPT interface. Run the Python file or execute the notebook cells in order.

Experiment settings, such as the number of demonstrations, random seed, and uncertainty threshold, can be adjusted in the code.

## Outputs

The code reports candidate formulas, nominal and worst-case satisfaction probabilities, selected specifications (winners), and computation times.

All model parameters and demonstrations are synthetic.
