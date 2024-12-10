from __future__ import annotations
import math
import pandas as pd
from leap.utils import get_data_path
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy

logger = get_logger(__name__)


class Birth:
    """A class containing information about projected birth rates.

    Attributes:
        estimate: A grouped data frame giving the projected number of births in a given
            province, grouped by year. It contains the following columns:
                * ``province``: A string indicating the province abbreviation, e.g. "BC".
                  For all of Canada, set province to "CA".
                * ``N``: integer, estimated number of births for that year.
                * ``prop_male``: proportion of births which are male, a number in ``[0, 1]``.
                * ``projection_scenario``: Population growth type, one of:
                    ["past", "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", FA", "SA"].
                    See `StatCan <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
                * ``N_relative``: number of births relative to the first year of the simulation.
            See ``master_birth_estimate.csv``.
        initial_population: A data frame giving the population for the first year of the simulation:
                * ``year``: integer year the range 2000 - 2065.
                * ``age``: integer age.
                * ``province``: a string indicating the province abbreviation, e.g. "BC".
                  For all of Canada, set province to "CA".
                * ``n``: estimated number of people in that age category in a given year.
                * ``n_birth``: the number of people born that year.
                * ``prop``: the ratio of that age group to the newborn age group (age = 0).
                * ``prop_male``: proportion of people in that age group who are male, a
                  number in [0, 1].
                * ``projection_scenario``: Population growth type, one of:
                  ``["past", "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", FA", "SA"]``.
                  See `StatCan <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
            See ``master_population_initial_distribution.csv``.
    """
    def __init__(
        self,
        starting_year: int | None = None,
        province: str | None = None,
        population_growth_type: str | None = None,
        max_age: int = 111,
        estimate: DataFrameGroupBy | None = None,
        initial_population: pd.DataFrame | None = None
    ):
        if starting_year is not None and province is not None and population_growth_type is not None:
            self.estimate = self.load_birth_estimate(
                starting_year, province, population_growth_type
            )
            self.initial_population = self.load_population_initial_distribution(
                starting_year, province, population_growth_type, max_age
            )
        elif estimate is not None and initial_population is not None:
            self.estimate = estimate
            self.initial_population = initial_population
        else:
            raise ValueError(
                "Either starting_year, province, and population_growth_type or "
                "estimate and initial_population must be provided."
            )
    
    @property
    def estimate(self) -> DataFrameGroupBy:
        return self._estimate
    
    @estimate.setter
    def estimate(self, estimate: DataFrameGroupBy) -> None:
        self._estimate = estimate

    def load_birth_estimate(
        self, starting_year: int, province: str, population_growth_type: str
    ) -> DataFrameGroupBy:
        df = pd.read_csv(
            get_data_path("processed_data", "master_birth_estimate.csv")
        )
        df = df[
            (df["year"] >= starting_year) &
            (df["province"] == province) &
            ((df["projection_scenario"] == population_growth_type) |
             (df["projection_scenario"] == "past"))
        ]
        df["N_relative"] = df["N"] / df["N"].iloc[0]
        grouped_df = df.groupby("year")
        return grouped_df

    def load_population_initial_distribution(
        self, starting_year: int, province: str, population_growth_type: str, max_age: int
    ) -> pd.DataFrame:
        df = pd.read_csv(
            get_data_path("processed_data", "master_initial_pop_distribution_prop.csv")
        )
        df = df[
            (df["age"] <= max_age) &
            (df["year"] == starting_year) &
            (df["province"] == province) &
            ((df["projection_scenario"] == population_growth_type) |
             (df["projection_scenario"] == "past"))
        ]
        return df

    def get_initial_population_indices(self, num_births: int) -> list[int]:
        """Get the indices for the agents from the initial population table, weighted by age.

        Examples:
            For example, if the number of births is 2, and we have the following
            initial population table:

            .. code-block::

                age | prop | ...
                ----------------
                0     1.0    ...
                1     2.0    ...
                2     0.5    ...

            then we will return the following:

            .. code-block::

                [1, 1, 2, 2, 2, 2, 3]

        Args:
            num_births: Number of births.

        Returns:
            The indices for the initial population table.
        """
        num_agents_per_age_group = [
            int(round(prop * num_births)) for prop in self.initial_population["prop"]
        ]
        initial_population_indices = []
        for age_index, num_agents in enumerate(num_agents_per_age_group):
            initial_population_indices.extend([age_index] * num_agents)
        return initial_population_indices

    def get_num_newborn(self, num_births_initial: int, year: int) -> int:
        """Get the number of births in a given year.

        Args:
            num_births_initial: Number of births in the initial year of the simulation.
            year: The calendar year.

        Returns:
            The number of births for the given year.
        """
        num_new_born = int(
            math.ceil(
                num_births_initial * self.estimate.get_group((year))["N_relative"] # type: ignore
            ) 
        ) 
        return num_new_born
