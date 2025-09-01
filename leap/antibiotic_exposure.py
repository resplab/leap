from __future__ import annotations
import copy
import pandas as pd
import numpy as np
from leap.utils import get_data_path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.utils import Sex


class AntibioticExposure:
    """A class containing information about antibiotic use."""

    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        mid_trends: DataFrameGroupBy | None = None
    ):
        if config is not None:
            self.parameters = config["parameters"]
        elif parameters is not None:
            self.parameters = parameters
        else:
            raise ValueError("Either config dict or parameters must be provided.")

        if mid_trends is None:
            self.mid_trends = self.load_abx_mid_trends()
        else:
            self.mid_trends = mid_trends

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:

            * ``β0``: (float); the constant parameter when computing μ.
            * ``βyear``: (float); the parameter to be multiplied by the agent's birth year for
              computing ``μ``.
            * ``β2005``: (float); an added constant parameter if the agent's birth year > 2005 for
              computing ``μ``. This is to factor in the antibiotic stewardship program that was
              introduced in BC in 2005.
            * ``βsex``: (float); the parameter to be multiplied by the agent's sex when computing μ.
            * ``θ``: int, the number of successes (the r parameter) in the negative binomial
              distribution.
            * ``β2005_year``: (float); If the agent's birth year is ``> 2005``, ``β2005_year``
              will be multiplied by the birth year when computing ``μ``. This is to factor in the
              antibiotic stewardship program that was introduced in BC in 2005.
            * ``fixyear``: (int | None); If present, replaces the ``year`` parameter when
              computing the probability for the negative binomial distribution.
            * ``βfloor``: (float); the minimum value of ``μ``.

        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["β0", "βyear", "β2005", "βsex", "θ", "β2005_year", "fixyear", "βfloor"]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"Key {key} not found in parameters.")
        self._parameters = copy.deepcopy(parameters)

    @property
    def mid_trends(self) -> DataFrameGroupBy:
        """A set of dataframes grouped by year and sex.

        Each entry is a dataframe with a single row with the following columns:

        * ``year (int)``: The calendar year, e.g. ``2024``.
        * ``sex``: 0 = female, 1 = male
        * ``rate (float)``: The average number of courses of antibiotics prescribed during
          infancy, per person.
        """
        return self._mid_trends
    
    @mid_trends.setter
    def mid_trends(self, mid_trends: DataFrameGroupBy):
        self._mid_trends = mid_trends

    def __copy__(self):
        return AntibioticExposure(
            parameters=self.parameters, mid_trends=self.mid_trends
        )

    def __deepcopy__(self):
        return AntibioticExposure(
            parameters=copy.deepcopy(self.parameters),
            mid_trends=copy.deepcopy(self.mid_trends)
        )

    def copy(self, deep: bool = True):
        if deep:
            return self.__deepcopy__()
        else:
            return self.__copy__()

    def load_abx_mid_trends(self):
        """Load the antibiotic mid trends table.

        Returns:
            A set of data frames grouped by year and sex.

            Each entry is a DataFrame with a single row with the following columns:

            * ``year (int)``: The calendar year, e.g. ``2024``.
            * ``sex``: 0 = female, 1 = male
            * ``rate (float)``: The average number of courses of antibiotics prescribed during
              infancy, per person.
        """
        df = pd.read_csv(get_data_path("processed_data/midtrends.csv"))
        grouped_df = df.groupby(["year", "sex"])
        return grouped_df

    def compute_num_antibiotic_use(self, sex: Sex | int, birth_year: int) -> int:
        """Compute the number of courses of antibiotics used during the first year of life.

        Args:
            sex: Sex of agent, 1 = male, 0 = female.
            birth_year: The year the agent (person) was born.

        Returns:
            The number of courses of antibiotics used during the first year of life.

        Examples:

            >>> from leap.antibiotic_exposure import AntibioticExposure
            >>> from leap.utils import get_data_path
            >>> import json
            >>> with open(get_data_path("processed_data/config.json"), "r") as file:
            ...     config = json.load(file)["antibiotic_exposure"]
            >>> antibiotic_exposure = AntibioticExposure(
            ...     config=config
            ... )
            >>> n_abx = antibiotic_exposure.compute_num_antibiotic_use(sex=1, birth_year=2000)

        """
        if birth_year < 2001:
            p = self.compute_probability(sex=sex, year=2000)
        elif self.parameters["fixyear"] is not None:
            if isinstance(self.parameters["fixyear"], (int, float)):
                p = self.compute_probability(
                    sex=sex,
                    year=int(self.parameters["fixyear"])
                )
            else:
                μ = max(
                    self.mid_trends.get_group((birth_year, int(sex)))["rate"].iloc[0],
                    self.parameters["βfloor"]
                )
                p = self.parameters["θ"] / (self.parameters["θ"] + μ)
        else:
            p = self.compute_probability(
                sex=sex,
                year=birth_year
            )
        r = self.parameters["θ"]
        return np.random.negative_binomial(r, p)

    def compute_probability(self, sex: Sex | int, year: int) -> float:
        """Compute the probability of antibiotic exposure for a given year and sex.

        Args:
            sex: Sex of agent, 1 = male, 0 = female.
            year: The calendar year, e.g. 2024.

        Returns:
            The ``p`` parameter for the Negative Binomial distribution, used to calculate
            the number of courses of antibiotics used during the first year of life.

        Examples:

            >>> from leap.antibiotic_exposure import AntibioticExposure
            >>> parameters = {
            ...     "β0": -100000,
            ...     "βyear": -0.01,
            ...     "βsex": -1,
            ...     "θ": 500,
            ...     "fixyear": None,
            ...     "βfloor": 0.0,
            ...     "β2005": 1,
            ...     "β2005_year": 1
            ... }
            >>> antibiotic_exposure = AntibioticExposure(
            ...     parameters=parameters
            ... )
            >>> antibiotic_exposure.compute_probability(sex=1, year=2000)
            1.0
        """
        η = (
            self.parameters["β0"] +
            self.parameters["βsex"] * int(sex) +
            self.parameters["βyear"] * year +
            self.parameters["β2005"] * (year > 2005) +
            self.parameters["β2005_year"] * (year > 2005) * year
        )

        μ = max(np.exp(η), self.parameters["βfloor"] / 1000)
        return float(self.parameters["θ"] / (self.parameters["θ"] + μ))
