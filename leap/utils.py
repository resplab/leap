import numpy as np
import pathlib
import math
import uuid
import importlib.resources as pkg_resources
from typing import Callable


LEAP_PATH = pathlib.Path(__file__).parents[1].absolute()
PROCESSED_DATA_PATH = pathlib.Path(LEAP_PATH, "processed_data")


class UUID4:
    def __init__(self, uuid4: uuid.UUID = uuid.uuid4(), short: str = str(uuid.uuid4())[-6:]):
        self.uuid4 = uuid4
        self.short = short


def round_number(x: float, digits: int = 0, sigdigits: int | None = None) -> float:
    """Rounds a number to number of significant figures or digits.

    https://mattgosden.medium.com/rounding-to-significant-figures-in-python-2415661b94c3

    Args:
        x (float): The number to be rounded.
        digits (int): The number of decimal places to round to.
        sigdigits (int): The number of significant figures to round to.

    Returns:
        float: The rounded number.
    """
    if sigdigits is not None:
        x = float(x)
        if x == 0:
            return round(x, sigdigits - 1)
        else:
            return round(x, -int(math.floor(math.log10(abs(x)))) + (sigdigits - 1))
    else:
        return round(x, digits)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    return np.log(x / (1 - x))


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


def get_data_path(file_name: str) -> str:
    """Get the full path to a file in the tests/data directory.

    Args:
        file_name: The name of the file.

    Returns:
        The full path to the file in the tests/data directory.
    """

    package_path = str(pkg_resources.files("tests.data").joinpath(file_name))
    return package_path


class Sex:
    def __init__(self, value: str | int | bool):
        if isinstance(value, str):
            if value == "M":
                value_str = "M"
                value_int = 1
                value_bool = True
            elif value == "F":
                value_str = "F"
                value_int = 0
                value_bool = False
            else:
                raise ValueError(f"sex must be 'M' or 'F', received {value}")
        elif isinstance(value, int) or isinstance(value, np.int64):
            if value == 1:
                value_str = "M"
                value_int = 1
                value_bool = True
            elif value == 0:
                value_str = "F"
                value_int = 0
                value_bool = False
            else:
                raise ValueError(f"sex must be 0 or 1, received {value}")
        elif isinstance(value, bool):
            if value:
                value_str = "M"
                value_int = 1
                value_bool = True
            else:
                value_str = "F"
                value_int = 0
                value_bool = False
        else:
            raise TypeError(f"sex must be str, int, or bool, received {type(value)}")
        
        self._value_str = value_str
        self._value_int = value_int
        self._value_bool = value_bool

    def __str__(self) -> str:
        return self._value_str
    
    def __int__(self) -> int:
        return self._value_int
    
    def __bool__(self) -> bool:
        return self._value_bool
    
    def __eq__(self, value: str | int | bool) -> bool:
        if isinstance(value, str):
            return self._value_str == value
        elif isinstance(value, int):
            return self._value_int == value
        elif isinstance(value, bool):
            return self._value_bool == value