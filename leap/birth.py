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
    """A class containing information about projected birth rates."""

    def __init__(
        self,
        starting_year: int | None = None,
        province: str | None = None,
        population_growth_type: str | None = None,
        max_age: int = 111,
        estimate: DataFrameGroupBy | None = None,
        initial_population: pd.DataFrame | None = None
    ):
        if estimate is None:
            if starting_year is None or province is None or population_growth_type is None:
                raise ValueError(
                    "Either starting_year, province, and population_growth_type or "
                    "estimate must be provided."
                )
            self.estimate = self.load_birth_estimate(
                starting_year, province, population_growth_type
            )
        else:
            self.estimate = estimate
        if initial_population is None:
            if starting_year is None or province is None or population_growth_type is None:
                raise ValueError(
                    "Either starting_year, province, and population_growth_type or "
                    "initial_population must be provided."
                )
            self.initial_population = self.load_population_initial_distribution(
                starting_year, province, population_growth_type, max_age
            )
        else:
            self.initial_population = initial_population
    
    @property
    def estimate(self) -> DataFrameGroupBy:
        """A grouped data frame giving the number of births in a given province, grouped by year.
        
        It contains the following columns:
            * ``province``: A string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``N``: integer, estimated number of births for that year.
            * ``prop_male``: The proportion of births which are male, a number in ``[0, 1]``.
            * ``projection_scenario``: Population growth type, one of:

              * ``past``: historical data
              * ``LG``: low-growth projection
              * ``HG``: high-growth projection
              * ``M1``: medium-growth 1 projection
              * ``M2``: medium-growth 2 projection
              * ``M3``: medium-growth 3 projection
              * ``M4``: medium-growth 4 projection
              * ``M5``: medium-growth 5 projection
              * ``M6``: medium-growth 6 projection
              * ``FA``: fast-aging projection
              * ``SA``: slow-aging projection

              See: `StatCan Projection Scenarios
              <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
            * ``N_relative``: The number of births relative to the first year of the simulation.

        See ``master_birth_estimate.csv``.
        """
        return self._estimate
    
    @estimate.setter
    def estimate(self, estimate: DataFrameGroupBy):
        self._estimate = estimate

    @property
    def initial_population(self) -> pd.DataFrame:
        """A data frame giving the population for the first year of the simulation.

        It contains the following columns:
            * ``year``: Integer year the range ``2000 - 2065``.
            * ``age``: Integer age.
            * ``province``: A string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``n``: Estimated number of people in that age category in a given year.
            * ``n_birth``: The number of people born that year.
            * ``prop``: The ratio of that age group to the newborn age group (age = 0).
            * ``prop_male``: The proportion of people in that age group who are male, a
              number in ``[0, 1]``.
            * ``projection_scenario``: Population growth type, one of:

              * ``past``: historical data
              * ``LG``: low-growth projection
              * ``HG``: high-growth projection
              * ``M1``: medium-growth 1 projection
              * ``M2``: medium-growth 2 projection
              * ``M3``: medium-growth 3 projection
              * ``M4``: medium-growth 4 projection
              * ``M5``: medium-growth 5 projection
              * ``M6``: medium-growth 6 projection
              * ``FA``: fast-aging projection
              * ``SA``: slow-aging projection

              See: `StatCan Projection Scenarios
              <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.

        See ``master_population_initial_distribution.csv``.
        """
        return self._initial_population
    
    @initial_population.setter
    def initial_population(self, initial_population: pd.DataFrame):
        self._initial_population = initial_population

    def load_birth_estimate(
        self, starting_year: int, province: str, population_growth_type: str
    ) -> DataFrameGroupBy:
        """Load the data from ``master_birth_estimate.csv``.
        
        Args:
            starting_year: The year for the data to start at. Must be between ``2000-2065``.
            province: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
            population_growth_type: Population growth type, one of:

                * ``past``: historical data
                * ``LG``: low-growth projection
                * ``HG``: high-growth projection
                * ``M1``: medium-growth 1 projection
                * ``M2``: medium-growth 2 projection
                * ``M3``: medium-growth 3 projection
                * ``M4``: medium-growth 4 projection
                * ``M5``: medium-growth 5 projection
                * ``M6``: medium-growth 6 projection
                * ``FA``: fast-aging projection
                * ``SA``: slow-aging projection

                See: `StatCan Projection Scenarios
                <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
        
        Returns:
            A grouped dataframe containing the number of births for a given province and
            projection scenario, grouped by year. Each group contains the following columns:

            * ``year``: Integer year the range ``2000 - 2065``.
            * ``province``: A string indicating the province abbreviation, e.g. ``"BC"``.
            * ``N``: Integer, estimated number of births for that year.
            * ``N_relative``: The number of births relative to the first year of the simulation.
            * ``prop_male``: The proportion of births which are male, a number in ``[0, 1]``.
            * ``projection_scenario``: Population growth type, one of:

              * ``past``: historical data
              * ``LG``: low-growth projection
              * ``HG``: high-growth projection
              * ``M1``: medium-growth 1 projection
              * ``M2``: medium-growth 2 projection
              * ``M3``: medium-growth 3 projection
              * ``M4``: medium-growth 4 projection
              * ``M5``: medium-growth 5 projection
              * ``M6``: medium-growth 6 projection
              * ``FA``: fast-aging projection
              * ``SA``: slow-aging projection

              See: `StatCan Projection Scenarios
              <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
        """

        df = pd.read_csv(
            get_data_path("processed_data/master_birth_estimate.csv")
        )
        df = df[
            (df["year"] >= starting_year) &
            (df["province"] == province) &
            ((df["projection_scenario"] == population_growth_type) |
             (df["projection_scenario"] == "past"))
        ]
        df["N_relative"] = df["N"] / df.loc[df["year"] == starting_year]["N"].values[0]
        grouped_df = df.groupby("year")
        return grouped_df

    def load_population_initial_distribution(
        self, starting_year: int, province: str, population_growth_type: str, max_age: int
    ) -> pd.DataFrame:
        """Load the data from ``master_initial_pop_distribution_prop.csv``.
        
        Args:
            starting_year: The year for the data to start at. Must be between ``2000-2065``.
            province: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
            population_growth_type: Population growth type, one of:

                * ``past``: historical data
                * ``LG``: low-growth projection
                * ``HG``: high-growth projection
                * ``M1``: medium-growth 1 projection
                * ``M2``: medium-growth 2 projection
                * ``M3``: medium-growth 3 projection
                * ``M4``: medium-growth 4 projection
                * ``M5``: medium-growth 5 projection
                * ``M6``: medium-growth 6 projection
                * ``FA``: fast-aging projection
                * ``SA``: slow-aging projection

                See: `StatCan Projection Scenarios
                <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
            max_age: The maximum age to include in the data.
        
        Returns:
            A dataframe containing the population for the first year of the simulation.
        """

        df = pd.read_csv(
            get_data_path("processed_data/master_initial_pop_distribution_prop.csv")
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

        Args:
            num_births: Number of births.

        Returns:
            The indices for the initial population table.

        Examples:

            >>> from leap.birth import Birth
            >>> from leap.utils import get_data_path
            >>> import pandas as pd
            >>> initial_population = pd.DataFrame({
            ...     "age": [0, 1, 2],
            ...     "prop": [1.0, 2.0, 0.5]
            ... })
            >>> birth = Birth(
            ...     starting_year=2000,
            ...     province="CA",
            ...     population_growth_type="LG",
            ...     initial_population=initial_population
            ... )
            >>> birth.get_initial_population_indices(num_births=2)
            [0, 0, 1, 1, 1, 1, 2]

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

        Examples:

            >>> from leap.birth import Birth
            >>> from leap.utils import get_data_path
            >>> birth = Birth(
            ...     starting_year=2000,
            ...     province="CA",
            ...     population_growth_type="LG"
            ... )
            >>> birth.get_num_newborn(num_births_initial=100, year=2000)
            100

            >>> birth.get_num_newborn(num_births_initial=100, year=2001)
            97

        """
        num_new_born = int(
            math.ceil(
                num_births_initial * self.estimate.get_group((year))["N_relative"] # type: ignore
            ) 
        ) 
        return num_new_born
