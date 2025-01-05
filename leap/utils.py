import numpy as np
import pathlib
import math
import os
import uuid
import importlib.resources as pkg_resources
from typing import Callable
from leap.logger import get_logger

logger = get_logger(__name__)


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
        x: The number to be rounded.
        digits: The number of decimal places to round to.
        sigdigits: The number of significant figures to round to.

    Returns:
        The rounded number.

    Examples:

        >>> round_number(2.932, digits=2)
        2.93
        >>> round_number(2.932, sigdigits=2)
        2.9
    """
    if sigdigits is not None:
        x = float(x)
        if x == 0:
            return round(x, sigdigits - 1)
        else:
            return round(x, -int(math.floor(math.log10(abs(x)))) + (sigdigits - 1))
    else:
        return round(x, digits)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-x))


def logit(x: float | np.ndarray) -> float | np.ndarray:
    return np.log(x / (1 - x))


def compute_ordinal_regression(
    η: float,
    θ: float | list[float],
    prob_function: Callable = sigmoid
) -> list[float]:
    """Compute the probability that ``y = k`` for each value of ``k``.

    The probability is given by ordinal regression:

    .. math::

        P(y <= k) &= σ(θ_k - η) \\\\
        P(y == k) &= P(y <= k) - P(y < k + 1) \\\\
                  &= σ(θ_k - η) - σ(θ_{(k+1)} - η)

    Args:
        η: The weight for the regression.
        θ: Either a single value or an array of values for the threshold parameter.
        prob_function: A function to apply; default is the sigmoid function.

    Returns:
        A vector with the probability of each value of ``k``.
        
        For example:

        .. code-block::

            k=1 | k=2  | k=2
            0.2 | 0.75 | 0.05

    Examples:

        >>> compute_ordinal_regression(0, 0.5)
        [np.float64(0.6224593312018546), np.float64(0.3775406687981454)]
        >>> compute_ordinal_regression(0, [0.5, 1.5])
        [np.float64(0.6224593312018546), np.float64(0.19511514499178906), np.float64(0.18242552380635635)]
    """

    if isinstance(θ, float):
        θ = [-10**5, θ, 10**5]
    else:
        θ = [-10**5] + θ + [10**5]

    return [prob_function(θ[k + 1] - η) - prob_function(θ[k] - η) for k in range(len(θ) - 1)]


def get_data_path(data_path: str) -> pathlib.Path:
    """Get the full path to a data file or folder in the ``LEAP`` package.

    Args:
        data_path: The path to the file or folder.

    Returns:
        The full path to the file or folder.
    """

    if pathlib.Path(data_path).parts[0] == "tests":
        data_path = str(pathlib.Path(data_path).relative_to("tests/data"))
    elif pathlib.Path(data_path).parts[0] == "processed_data":
        data_path = str(pathlib.Path(data_path).relative_to("processed_data"))

    for data_folder in ["tests.data", "processed_data"]:
        try:
            package_path = str(pkg_resources.files(data_folder).joinpath(data_path))
            if os.path.isfile(package_path) or os.path.isdir(package_path):
                return pathlib.Path(package_path)
        except ModuleNotFoundError:
            pass
        
    raise FileNotFoundError(f"Path {data_path} not found.")


def check_file(file_path: str | pathlib.Path, ext: str):
    """Check if file is a valid file with correct extension.

    Args:
        file_path: The full path to the file.
        ext: A file extension, including the ``"."``, e.g. ``".csv"``.

    Raises:
        ValueError: If the file is not a valid file with the correct extension.
    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    if os.path.isfile(file_path):
        file_path = file_path.resolve()
        file_ext = file_path.suffix
        if file_ext != ext:
            logger.error(f"{file_path} has extension {file_ext}, must be {ext}.")
            raise ValueError(f"{file_path} has extension {file_ext}, must be {ext}.")
    else:
        raise ValueError(f"{file_path} is not a valid file.")


class Sex:
    """A class to handle different formats of the ``sex`` variable."""
    def __init__(self, value: str | int | bool):
        """Initialize the ``Sex`` class.
        
        Args:
            value: The value of the sex variable. Must be one of:
                ``"M"``, ``"F"``, ``1``, ``0``, ``True``, or ``False``.
        
        Examples:

            >>> sex = Sex("F")
            >>> str(sex)
            'F'
            >>> int(sex)
            0
            >>> bool(sex)
            False

            >>> sex = Sex(True)
            >>> str(sex)
            'M'
            >>> int(sex)
            1
            >>> bool(sex)
            True
        """
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