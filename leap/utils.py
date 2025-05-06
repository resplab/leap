import numpy as np
import pandas as pd
import pathlib
import math
import os
import uuid
from scipy.stats import logistic
import importlib.resources as pkg_resources
from typing import Callable
from leap.logger import get_logger

logger = get_logger(__name__)


LEAP_PATH = pathlib.Path(__file__).parents[1].absolute()


class UUID4:
    def __init__(
        self, uuid4: uuid.UUID | None = None, short: str | None = None
    ):
        if uuid4 is None:
            uuid4 = uuid.uuid4()
        if short is None:
            short = str(uuid4)[-6:]
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


def logit(x: float | np.ndarray) -> float | np.ndarray:
    return np.log(x / (1 - x))


def compute_ordinal_regression(
    η: float,
    θ: float | list[float],
    prob_function: Callable = logistic.cdf
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
        data_folder = "tests.data"
        data_path = str(pathlib.Path(data_path).relative_to("tests/data"))
    elif pathlib.Path(data_path).parts[0] == "processed_data":
        data_folder = "leap.processed_data"
        data_path = str(pathlib.Path(data_path).relative_to("processed_data"))
    elif pathlib.Path(data_path).parts[0] == "original_data":
        data_path = str(pathlib.Path(data_path).relative_to("original_data"))
        data_folder = "leap.original_data"

    package_path = str(pkg_resources.files(data_folder).joinpath(data_path))

    if os.path.isfile(package_path) or os.path.isdir(package_path):
        return pathlib.Path(package_path)
    else:
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


def check_cduid(cduid: int, df: pd.DataFrame):
    """Check if the ``CDUID`` is valid.

    Args:
        cduid: The census division unique identifier.
        df: The DataFrame to check.

    Raises:
        ValueError: If the ``CDUID`` is not valid.
    """
    if cduid not in df["CDUID"].unique():
        raise ValueError(f"cduid must be one of {df['cduid'].unique()}, received {cduid}")


def check_year(year: int, df: pd.DataFrame):
    """Check if the year is valid.

    Args:
        year: The minimum year.
        df: The DataFrame to check.

    Raises:
        ValueError: If the year is not valid.
    """
    if year < df["year"].min():
        raise ValueError(f"year must be >= {df['year'].min()}")
    elif year > df["year"].max():
        raise ValueError(f"year must be <= {df['year'].max()}")


def check_province(province: str):
    """Check if the province is valid.

    Args:
        province: The province abbreviation.

    Raises:
        ValueError: If the province is not valid.
    """

    PROVINCES = [
        "CA", "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"
    ]
    if province not in PROVINCES:
        raise ValueError(f"province must be one of {PROVINCES}, received {province}")


def check_projection_scenario(projection_scenario: str):
    """Check if the projection scenario is valid.

    Args:
        projection_scenario: The projection scenario abbreviation.

    Raises:
        ValueError: If the projection scenario is not valid.
    """

    PROJECTION_SCENARIOS = [
        "past", "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", "FA", "SA"
    ]
    if projection_scenario not in PROJECTION_SCENARIOS:
        raise ValueError(
            f"projection_scenario must be one of {PROJECTION_SCENARIOS}, "
            f"received {projection_scenario}"
        )


def convert_non_serializable(obj: np.ndarray | object) -> list | str:
    """Convert non-serializable objects into JSON-friendly formats.

    This function is intended to be used with ``json.dumps(..., default=convert_non_serializable)``.
    It handles cases where objects cannot be directly serialized into JSON:

    - NumPy arrays (``numpy.ndarray``) are converted to Python lists.
    - Other unsupported types are converted to their string representation.

    Args:
        obj: The object to be converted.

    Returns:
        A JSON-serializable equivalent of ``obj``.
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Default for unsupported types
    return str(obj)

def convert_time_unit_to_year_month(time_unit: int, starting_year: int) -> tuple[int, int]:
    """
    Converts a 1-indexed time unit (months since start) into the corresponding
    calendar year and month, also 1-indexed.
    
    Args:
        time_unit: the time index in months (1 = Jan of starting year, 2 = Feb, …)
        starting_year: the initial year of the simulation
        
    Returns:
        A (year, month) tuple, both 1-indexed.
        
    Examples:
        >>> convert_time_unit_to_year_month(1, 2001)
        (2001, 1)   # January 2001
        >>> convert_time_unit_to_year_month(12, 2001)
        (2001, 12)  # December 2001
        >>> convert_time_unit_to_year_month(13, 2001)
        (2002, 1)   # January 2002
        >>> convert_time_unit_to_year_month(50, 2001)
        (2005, 2)   # February 2005
    """
    year  = starting_year + (time_unit - 1) // 12
    month = ((time_unit - 1) % 12) + 1
    return year, month

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
