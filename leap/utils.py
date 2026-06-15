from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import functools
import time
import pathlib
import math
import os
import sys
import uuid
from tqdm import tqdm
from scipy.stats import logistic
import importlib.resources as pkg_resources
from typing import Callable, Tuple, TYPE_CHECKING
import multiprocessing as mp
from leap.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

logger = get_logger(__name__)


LEAP_PATH = pathlib.Path(__file__).parents[1].absolute()

PROJECTION_SCENARIOS_FUTURE = (
    "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", "FA", "SA"
)
PROJECTION_SCENARIOS = ("past",) + PROJECTION_SCENARIOS_FUTURE

PROVINCE_MAP = {
    "Canada": "CA",
    "British Columbia": "BC",
    "Alberta": "AB",
    "Saskatchewan": "SK",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
    "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Prince Edward Island": "PE",
    "Yukon": "YT",
    "Northwest Territories": "NT",
    "Nunavut": "NU"
}

MORTALITY_SCENARIOS = ("HM", "MM", "LM")



def get_chunk_indices(
    n: int, chunk_size: int = 10
) -> list[Tuple[int, int]]:
    """Get the indices for chunks of size ``chunk_size`` from a total of ``n`` items.

    Args:
        n: The total number of items.
        chunk_size: The size of each chunk.

    Returns:
        A list of tuples, where each tuple contains the start and end indices for each chunk.

    Examples:

        >>> get_chunk_indices(n=25, chunk_size=10)
        [(0, 10), (10, 20), (20, 25)]
    """
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def create_process_bars(
    chunk_indices: list[Tuple[int, int]],
    position_offset: int = 0
) -> list[tqdm]:
    """Create a list of tqdm progress bars for each process.

    We have a main job bar that shows the total progress, and a sub-bar for each process:

    Job Bar: 96% [=================== ] | 240/250
    Process 1: 100% [=================] | 63/63
    Process 2: 95%  [================ ] | 60/63
    Process 3: 100% [=================] | 63/63
    Process 4: 89%  [==============   ] | 54/61
    

    Args:
        chunk_indices: A list of tuples containing the start and end indices for each chunk.
        position_offset: The position offset for the progress bars in the tqdm display.

    Returns:
        A list of tqdm progress bars, one for each chunk.
    """
    return [
        tqdm(
            total=(chunk_indices[i][1] - chunk_indices[i][0]),
            position=i + position_offset,
            desc=f"| -- Process {i}",
            leave=False,
            file=sys.stdout
        ) for i in range(len(chunk_indices))
    ]



def timer(log_level: int = 20) -> Callable:
    def timer_decorator(func: Callable) -> Callable:
        """Print the runtime of the decorated function"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            if log_level == 25:
                logger.message(f"Finished {func.__name__}() in {run_time:.6f} seconds")
            elif log_level == 20:
                logger.info(f"Finished {func.__name__}() in {run_time:.6f} seconds")
            return value
        return wrapper_timer
    return timer_decorator


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
        [0.6224593312018546, 0.3775406687981454]
        >>> compute_ordinal_regression(0, [0.5, 1.5])
        [0.6224593312018546, 0.19511514499178906, 0.18242552380635635]
    """

    if isinstance(θ, float):
        θ = [-10**5, θ, 10**5]
    else:
        θ = [-10**5] + θ + [10**5]

    return [float(prob_function(θ[k + 1] - η) - prob_function(θ[k] - η)) for k in range(len(θ) - 1)]


def get_data_path(data_path: str | pathlib.Path, mkdirs: bool = False) -> pathlib.Path:
    """Get the full path to a data file or folder in the ``LEAP`` package.

    Args:
        data_path: The path to the file or folder.
        mkdirs: Whether to create the directories if they don't exist.

    Returns:
        The full path to the file or folder.
    """

    data_path = pathlib.Path(data_path)
    if data_path.parts[0] == "tests":
        data_folder = "tests.data"
        data_path = data_path.relative_to("tests/data")
    elif data_path.parts[0] == "processed_data":
        data_folder = "leap.processed_data"
        data_path = data_path.relative_to("processed_data")
    elif data_path.parts[0] == "data_generation":
        data_folder = "leap.data_generation"
        data_path = data_path.relative_to("data_generation")
    elif data_path.parts[0] == "original_data":
        data_path = data_path.relative_to("original_data")
        data_folder = "leap.original_data"
    else:
        raise FileNotFoundError(f"Path {data_path} not found.")
  
    if str(data_path) == ".":
        package_path = pkg_resources.files(data_folder)
    else:
        package_path = pkg_resources.files(data_folder).joinpath(str(data_path))

    if package_path.is_file() or package_path.is_dir():
        with pkg_resources.as_file(package_path) as f:
            path = pathlib.Path(f)
        return path
    else:
        if mkdirs:
            with pkg_resources.as_file(package_path) as f:
                path = pathlib.Path(f)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path
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


def check_timepoint(timepoint: dt.datetime, df: pd.DataFrame):
    """Check if the timepoint is valid.

    Args:
        timepoint: The timepoint to check.
        df: The DataFrame to check.

    Raises:
        ValueError: If the timepoint is not valid.
    """
    if timepoint < df["timepoint"].min():
        raise ValueError(f"timepoint must be >= {df['timepoint'].min()}")
    elif timepoint > df["timepoint"].max():
        raise ValueError(f"timepoint must be <= {df['timepoint'].max()}")


def check_province(province: str):
    """Check if the province is valid.

    Args:
        province: The province abbreviation.

    Raises:
        ValueError: If the province is not valid.
    """

    if province not in PROVINCE_MAP.values():
        raise ValueError(
            f"province must be one of {list(PROVINCE_MAP.values())}, received {province}"
        ) 


def check_projection_scenario(projection_scenario: str):
    """Check if the projection scenario is valid.

    Args:
        projection_scenario: The projection scenario abbreviation.

    Raises:
        ValueError: If the projection scenario is not valid.
    """

    if projection_scenario not in PROJECTION_SCENARIOS:
        raise ValueError(
            f"projection_scenario must be one of {PROJECTION_SCENARIOS}, "
            f"received {projection_scenario}"
        )


def poly(
    x: list[float] | np.ndarray | float,
    degree: int = 1,
    alpha: list[float] | np.ndarray | None = None,
    norm2: list[float] | np.ndarray | None = None,
    orthogonal: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a polynomial basis for a vector.

    See `Orthogonal polynomial regression in Python
    <https://davmre.github.io/blog/python/2013/12/15/orthogonal_poly/>`_ for more
    information on this function.
    
    Args:
        x: The vector to generate the polynomial basis for.
        degree: The degree of the polynomial.
        orthogonal: Whether to generate an orthogonal polynomial basis.
        
    Returns:
        The polynomial basis, as a 2D Numpy array. If ``orthogonal`` is ``True``, the function
        will return a tuple of three Numpy arrays: the orthogonal polynomial basis, the ``alpha``
        values, and the ``norm2`` values.

    Examples:

        >>> poly([1, 2, 3], degree=2) # doctest: +NORMALIZE_WHITESPACE
        array([[1, 1],
               [2, 4],
               [3, 9]])

        >>> poly([1, 2, 3], degree=2, orthogonal=True) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        (array([[-7.07106781e-01,  4.08248290e-01],
               [..., -8.16496581e-01],
               [ 7.07106781e-01,  4.08248290e-01]]), array([2., 2.]), array([3.        , 2.        , 0.66666667]))
    """
    n = degree + 1
    x = np.asarray(x).flatten()
    if alpha is not None and norm2 is not None:
        Z = np.empty((len(x), n))
        Z[:,0] = 1
        if degree > 0:
            Z[:, 1] = x - alpha[0]
        if degree > 1:
            for i in np.arange(1, degree):
                Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
        Z /= np.sqrt(norm2)
        return Z[:, 1:]
    else:
        if(degree >= len(np.unique(x))):
            raise ValueError("'degree' must be less than number of unique points")
        if orthogonal:
            xbar = np.mean(x)
            x = x - xbar
            X = np.vander(x, n, increasing=True)
            Q, R = np.linalg.qr(X)

            Z = np.diag(np.diag(R))
            raw = np.dot(Q, Z)

            norm2 = np.sum(raw**2, axis=0)
            alpha = (
                np.sum((raw**2) * np.reshape(x, (-1, 1)), axis=0)/norm2 + 
                xbar
            )[:degree]
            Z = raw / np.sqrt(norm2)
            return Z[:, 1:], alpha, norm2
        else:
            X = np.vander(x, n, increasing=True)
            return X[:, 1:]


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


def get_time_delta_tag(time_delta: dt.timedelta | relativedelta) -> str:
    if isinstance(time_delta, dt.timedelta):
        return f"time_delta_{time_delta.days}"
    elif isinstance(time_delta, relativedelta):
        days = 0
        if time_delta.years > 0:
            days += time_delta.years * 365
        if time_delta.months > 0:
            days += time_delta.months * 30
        if time_delta.days > 0:
            days += time_delta.days
        return f"time_delta_{days}"
    else:
        raise TypeError("time_delta must be a timedelta or relativedelta.")


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


def date_range(
    start: dt.datetime, stop: dt.datetime, step: dt.timedelta | relativedelta | TimeDelta
) -> Generator[dt.datetime, None, None]:
    current = start
    while current < stop:
        yield current
        current += step

class TimeDelta(relativedelta):
    def __init__(
        self,
        dt1=None,
        dt2=None,
        years: int = 0,
        months: int = 0,
        days: int = 0,
        leapdays: int = 0,
        weeks: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
        year=None,
        month=None,
        day=None,
        weekday=None,
        yearday=None,
        nlyearday=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        iso_string: str | None = None,
        rd: relativedelta | None = None,
        td: dt.timedelta | None = None
    ):
        if rd is not None:
            super().__init__(
                years=rd.years,
                months=rd.months,
                days=rd.days,
                leapdays=rd.leapdays,
                weeks=rd.weeks,
                hours=rd.hours,
                minutes=rd.minutes,
                seconds=rd.seconds,
                microseconds=rd.microseconds,
                year=rd.year,
                month=rd.month,
                day=rd.day,
                weekday=rd.weekday,
                hour=rd.hour,
                minute=rd.minute,
                second=rd.second,
                microsecond=rd.microsecond
            )
        elif td is not None:
            super().__init__(
                days=td.days,
                seconds=td.seconds,
                microseconds=td.microseconds
            )
        elif iso_string is not None:
            self.iso_string = iso_string
        else:
            super().__init__(
                dt1=dt1,
                dt2=dt2,
                years=years,
                months=months,
                days=days,
                leapdays=leapdays,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
                year=year,
                month=month,
                day=day,
                weekday=weekday,
                yearday=yearday,
                nlyearday=nlyearday,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond
            )

    @property
    def iso_string(self) -> str | None:
        """The ISO 8601 string representation of the time interval, e.g. ``"P1Y2M3DT4H5M6S"``."""
        return self._iso_string

    @iso_string.setter
    def iso_string(self, iso_string: str | None):
        if iso_string is not None:
            if not iso_string.startswith("P"):
                raise ValueError(f"iso_string must be in ISO 8601 format, received {iso_string}")

            iso_string = iso_string[1:]
            date_part, time_part = iso_string.split("T") if "T" in iso_string else (iso_string, "")
 
            if len(date_part.split("Y")) == 2:
                years = int(date_part.split("Y")[0])
            else:
                years = 0
            if len(date_part.split("M")) == 2:
                months = int(date_part.split("M")[0].split("Y")[-1])
            else:
                months = 0
            if len(date_part.split("D")) == 2:
                days = int(date_part.split("D")[0].split("M")[-1].split("Y")[-1])
            else:
                days = 0
            if len(time_part.split("H")) == 2:
                hours = int(time_part.split("H")[0])
            else:
                hours = 0
            if len(time_part.split("M")) == 2:
                minutes = int(time_part.split("M")[0].split("H")[-1])
            else:
                minutes = 0
            if len(time_part.split("S")) == 2:
                seconds = int(time_part.split("S")[0].split("M")[-1].split("H")[-1])
            else:
                seconds = 0

            super().__init__(
                years=years,
                months=months,
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds
            )
        self._iso_string = iso_string
    
    def __lt__(self, other: TimeDelta | dt.timedelta | relativedelta) -> bool:
        if isinstance(other, dt.timedelta) or isinstance(other, TimeDelta):
            return self.total_seconds() < other.total_seconds()
        elif isinstance(other, relativedelta):
            total_seconds = (
                other.years * 365 * 24 * 3600 + 
                other.months * 30 * 24 * 3600 + 
                other.days * 24 * 3600 + 
                other.hours * 3600 + 
                other.minutes * 60 + 
                other.seconds + 
                other.microseconds / 1e6
            )
            return self.total_seconds() < total_seconds
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        
    def __le__(self, other: TimeDelta | dt.timedelta | relativedelta) -> bool:
        if isinstance(other, dt.timedelta) or isinstance(other, TimeDelta) or isinstance(other, relativedelta):
            return self.__lt__(other) or self.__eq__(other)
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        
    def __gt__(self, other: TimeDelta | dt.timedelta | relativedelta) -> bool:
        if isinstance(other, dt.timedelta) or isinstance(other, TimeDelta):
            return self.total_seconds() > other.total_seconds()
        elif isinstance(other, relativedelta):
            total_seconds = (
                other.years * 365 * 24 * 3600 + 
                other.months * 30 * 24 * 3600 + 
                other.days * 24 * 3600 + 
                other.hours * 3600 + 
                other.minutes * 60 + 
                other.seconds + 
                other.microseconds / 1e6
            )
            return self.total_seconds() > total_seconds
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        
    def __ge__(self, other: TimeDelta | dt.timedelta | relativedelta) -> bool:
        if isinstance(other, dt.timedelta) or isinstance(other, TimeDelta) or isinstance(other, relativedelta):
            return self.__gt__(other) or self.__eq__(other)
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        
    def __str__(self) -> str:
        return self.to_isoformat()
    
    def __truediv__(self, other: TimeDelta | dt.timedelta) -> float:
        if isinstance(other, TimeDelta) or isinstance(other, dt.timedelta):
            return self.total_seconds() / other.total_seconds()
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
    
    def __floordiv__(self, other: TimeDelta | dt.timedelta) -> int:
        if isinstance(other, TimeDelta) or isinstance(other, dt.timedelta):
            return int(self.total_seconds() // other.total_seconds())
        else:
            raise TypeError(f"Unsupported type for floor division: {type(other)}")
        
    def total_seconds(self) -> float:
        return (
            self.years * 365 * 24 * 3600 +
            self.months * 30 * 24 * 3600 +
            self.days * 24 * 3600 +
            self.hours * 3600 +
            self.minutes * 60 +
            self.seconds +
            self.microseconds / 1e6
        )
    
    def total_years(self) -> float:
        """Convert the total duration of time into units of years."""
        return self.total_seconds() / (365 * 24 * 3600)
    
    def to_isoformat(self) -> str:
        date_part = "".join([
            f"{self.years}Y" if self.years else "",
            f"{self.months}M" if self.months else "",
            f"{self.days}D" if self.days else "",
        ])
        time_part = "".join([
            f"{self.hours}H" if self.hours else "",
            f"{self.minutes}M" if self.minutes else "",
            f"{self.seconds}S" if self.seconds else "",
        ])
        return f"P{date_part}" + (f"T{time_part}" if time_part else "")
    
    def to_dateoffset(self) -> pd.DateOffset:
        return pd.DateOffset(
            years=self.years,
            months=self.months,
            days=self.days,
            hours=self.hours,
            minutes=self.minutes,
            seconds=self.seconds,
            microseconds=self.microseconds
        )


