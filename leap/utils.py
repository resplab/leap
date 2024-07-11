import numpy as np
import pathlib
from typing import Callable


LEAP_PATH = pathlib.Path(__file__).parents[1].absolute()
PROCESSED_DATA_PATH = pathlib.Path(LEAP_PATH, "processed_data")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_ordinal_regression(
    η: float,
    θ: float | list[float],
    prob_function: Callable = sigmoid
) -> list[float]:
    """Compute the probability that y = k for each value of k.

    The probability is given by ordinal regression:
        P(y <= k) = σ(θ_k - η)
        P(y == k) = P(y <= k) - P(y < k + 1)
                  = σ(θ_k - η) - σ(θ_(k+1) - η)

    Args:
        η (float): the weight for the regression.
        θ (float | list[float]): either a single value or an array of values for the
            threshold parameter.
        prob_function (function): A function to apply, default is the sigmoid function.

    Returns:
        list[float]: a vector with the probability of each value of k.
            For example:
            k=1 | k=2  | k=2
            0.2 | 0.75 | 0.05
    """

    if isinstance(θ, float):
        θ = [-10**5, θ, 10**5]
    else:
        θ = [-10**5] + θ + [10**5]

    return [prob_function(θ[k + 1] - η) - prob_function(θ[k] - η) for k in range(len(θ) - 1)]
