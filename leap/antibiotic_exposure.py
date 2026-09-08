from __future__ import annotations
import copy
from sqlite3 import Time
import time
import pandas as pd
import numpy as np
import datetime as dt
from leap.utils import get_data_path, get_time_delta_tag, TimeDelta
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.utils import Sex
    from dateutil.relativedelta import relativedelta


class AntibioticExposure:
    """A class containing information about antibiotic use."""

    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        data: DataFrameGroupBy | None = None,
        time_delta: dt.timedelta | relativedelta | TimeDelta = TimeDelta(years=1)
    ):
        if config is not None:
            self.parameters = config["parameters"]
        elif parameters is not None:
            self.parameters = parameters
        else:
            raise ValueError("Either config dict or parameters must be provided.")

        if data is None:
            self.data = self.load_abx_data(time_delta)
        else:
            self.data = data

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:

            * ``β0``: (float); the constant parameter when computing μ.
            * ``βtime``: (float); the parameter to be multiplied by the agent's birth timepoint for
              computing ``μ``.
            * ``β2005``: (float); an added constant parameter if the agent's birth year > 2005 for
              computing ``μ``. This is to factor in the antibiotic stewardship program that was
              introduced in BC in 2005.
            * ``βsex``: (float); the parameter to be multiplied by the agent's sex when computing μ.
            * ``θ``: int, the number of successes (the r parameter) in the negative binomial
              distribution.
            * ``β2005_time``: (float); If the agent's birth year is ``> 2005``, ``β2005_time``
              will be multiplied by the birth year when computing ``μ``. This is to factor in the
              antibiotic stewardship program that was introduced in BC in 2005.
            * ``fixyear``: (int | None); If present, replaces the ``year`` parameter when
              computing the probability for the negative binomial distribution.
            * ``βfloor``: (float); the minimum value of ``μ``.

        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["β0", "βtime", "β2005", "βsex", "θ", "β2005_time", "fixyear", "βfloor"]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"Key {key} not found in parameters.")
        self._parameters = copy.deepcopy(parameters)

    @property
    def data(self) -> DataFrameGroupBy:
        """A set of dataframes grouped by timepoint and sex.

        Each entry is a dataframe with a single row with the following columns:

        * ``timepoint (dt.datetime)``: The date and time, e.g. ``2024``.
        * ``sex``: 0 = female, 1 = male
        * ``n_abx_μ (float)``: The average number of courses of antibiotics prescribed during
          infancy, per person.
        """
        return self._data
    
    @data.setter
    def data(self, data: DataFrameGroupBy):
        self._data = data

    def __copy__(self):
        return AntibioticExposure(
            parameters=self.parameters, data=self.data
        )

    def __deepcopy__(self):
        return AntibioticExposure(
            parameters=copy.deepcopy(self.parameters),
            data=copy.deepcopy(self.data)
        )

    def copy(self, deep: bool = True):
        if deep:
            return self.__deepcopy__()
        else:
            return self.__copy__()

    def load_abx_data(self, time_delta: dt.timedelta | relativedelta | TimeDelta):
        """Load the antibiotic table.

        Returns:
            A set of data frames grouped by timepoint and sex.

            Each entry is a DataFrame with a single row with the following columns:

            * ``timepoint (dt.datetime)``: The date and time, e.g. ``2024``.
            * ``sex``: 0 = female, 1 = male
            * ``n_abx_μ (float)``: The average number of courses of antibiotics prescribed during
              infancy, per person.
        """
        time_delta_tag = get_time_delta_tag(time_delta)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/antibiotic_predictions.csv"),
            parse_dates=["timepoint"]
        )
        grouped_df = df.groupby(["timepoint", "sex"])
        return grouped_df

    def compute_num_antibiotic_use(self, sex: Sex | int, birth_year: int) -> int:
        """Compute the number of courses of antibiotics used during the first year of life.

        Args:
            sex: Sex of agent, 1 = male, 0 = female.
            birth_year: The timepoint (year) the agent (person) was born.

        Returns:
            The number of courses of antibiotics used during the first year of life.

        Examples:

            >>> from leap.antibiotic_exposure import AntibioticExposure
            >>> from leap.utils import get_data_path
            >>> import json
            >>> with open(get_data_path("processed_data/time_delta_365/config.json"), "r") as file:
            ...     config = json.load(file)["antibiotic_exposure"]
            >>> antibiotic_exposure = AntibioticExposure(
            ...     config=config
            ... )
            >>> n_abx = antibiotic_exposure.compute_num_antibiotic_use(sex=1, birth_year=2000)

        """
        if birth_year < 2001:
            p = self.compute_probability(sex=sex, timepoint=dt.datetime(2000, 1, 1))
        elif self.parameters["fixyear"] is not None:
            if isinstance(self.parameters["fixyear"], (int, float)):
                p = self.compute_probability(
                    sex=sex,
                    timepoint=dt.datetime(int(self.parameters["fixyear"]), 1, 1)
                )
            else:
                μ = max(
                    self.data.get_group((birth_year, int(sex)))["n_abx_μ"].iloc[0],
                    self.parameters["βfloor"]
                )
                p = self.parameters["θ"] / (self.parameters["θ"] + μ)
        else:
            p = self.compute_probability(
                sex=sex,
                timepoint=dt.datetime(birth_year, 1, 1)
            )
        r = self.parameters["θ"]
        return np.random.negative_binomial(r, p)

    def compute_probability(self, sex: Sex | int, timepoint: dt.datetime) -> float:
        """Compute the probability of antibiotic exposure for a given year and sex.

        Args:
            sex: Sex of agent, 1 = male, 0 = female.
            timepoint: The given timepoint (year) to compute the probability for, e.g. ``2024``.

        Returns:
            The ``p`` parameter for the Negative Binomial distribution, used to calculate
            the number of courses of antibiotics used during the first year of life.

        Examples:

            >>> from leap.antibiotic_exposure import AntibioticExposure
            >>> parameters = {
            ...     "β0": -100000,
            ...     "βtime": -0.01,
            ...     "βsex": -1,
            ...     "θ": 500,
            ...     "fixyear": None,
            ...     "βfloor": 0.0,
            ...     "β2005": 1,
            ...     "β2005_time": 1
            ... }
            >>> antibiotic_exposure = AntibioticExposure(
            ...     parameters=parameters
            ... )
            >>> antibiotic_exposure.compute_probability(sex=1, timepoint=dt.datetime(2000, 1, 1))
            1.0
        """
        η = (
            self.parameters["β0"] +
            self.parameters["βsex"] * int(sex) +
            self.parameters["βtime"] * timepoint.year +
            self.parameters["β2005"] * (timepoint.year > 2005) +
            self.parameters["β2005_time"] * (timepoint.year > 2005) * timepoint.year
        )

        μ = max(np.exp(η), self.parameters["βfloor"] / 1000)
        return float(self.parameters["θ"] / (self.parameters["θ"] + μ))
