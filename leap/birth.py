from __future__ import annotations
import math
import pandas as pd
import datetime as dt
from leap.utils import get_data_path, check_timepoint, check_province, check_projection_scenario, \
    get_time_delta_tag, TimeDelta
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from dateutil.relativedelta import relativedelta

logger = get_logger(__name__)


class Birth:
    """A class containing information about projected birth rates."""

    def __init__(
        self,
        min_timepoint: dt.datetime | None = None,
        province: str | None = None,
        projection_scenario: str | None = None,
        max_age: int = 111,
        estimate: DataFrameGroupBy | None = None,
        initial_population: pd.DataFrame | None = None,
        time_delta: dt.timedelta | relativedelta | TimeDelta = TimeDelta(years=1)
    ):
        if estimate is None:
            if min_timepoint is None or province is None or projection_scenario is None:
                raise ValueError(
                    "Either min_timepoint, province, and projection_scenario or "
                    "estimate must be provided."
                )
            self.estimate = self.load_birth_estimate(
                min_timepoint, province, projection_scenario, time_delta
            )
        else:
            self.estimate = estimate
        if initial_population is None:
            if min_timepoint is None or province is None or projection_scenario is None:
                raise ValueError(
                    "Either min_timepoint, province, and projection_scenario or "
                    "initial_population must be provided."
                )
            self.initial_population = self.load_population_initial_distribution(
                min_timepoint, province, projection_scenario, max_age, time_delta
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

        See ``birth_estimate.csv``.
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
        self,
        min_timepoint: dt.datetime,
        province: str,
        projection_scenario: str,
        time_delta: dt.timedelta | relativedelta | TimeDelta
    ) -> DataFrameGroupBy:
        """Load the data from ``birth_estimate.csv``.
        
        Args:
            min_timepoint: The year for the data to start at. Must be between ``2000-2065``.
            province: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
            projection_scenario: Population growth type, one of:

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

        time_delta_tag = get_time_delta_tag(time_delta)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/birth/birth_estimate.csv"),
            parse_dates=["timepoint"]
        )
        check_timepoint(min_timepoint, df)
        check_province(province)
        check_projection_scenario(projection_scenario)

        df = df[
            (df["timepoint"] >= min_timepoint) &
            (df["province"] == province) &
            ((df["projection_scenario"] == projection_scenario) |
             (df["projection_scenario"] == "past"))
        ]
        df["N_relative"] = df["N"] / df.loc[df["timepoint"] == min_timepoint]["N"].values[0]
        grouped_df = df.groupby("timepoint")
        return grouped_df

    def load_population_initial_distribution(
        self,
        min_timepoint: dt.datetime,
        province: str,
        projection_scenario: str,
        max_age: int,
        time_delta: dt.timedelta | relativedelta | TimeDelta
    ) -> pd.DataFrame:
        """Load the data from ``initial_population.csv``.
        
        Args:
            min_timepoint: The year for the data to start at. Must be between ``2000-2065``.
            province: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
            projection_scenario: Population growth type, one of:

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
        time_delta_tag = get_time_delta_tag(time_delta)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/birth/initial_population.csv"),
            parse_dates=["timepoint"]
        )

        check_timepoint(min_timepoint, df)
        check_province(province)
        check_projection_scenario(projection_scenario)

        df = df[
            (df["age"] <= max_age) &
            (df["timepoint"] == min_timepoint) &
            (df["province"] == province) &
            ((df["projection_scenario"] == projection_scenario) |
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
            >>> import datetime as dt
            >>> initial_population = pd.DataFrame({
            ...     "age": [0, 1, 2],
            ...     "prop": [1.0, 2.0, 0.5]
            ... })
            >>> birth = Birth(
            ...     min_timepoint=dt.datetime(2000, 1, 1),
            ...     province="CA",
            ...     projection_scenario="LG",
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

    def get_num_newborn(self, num_births_initial: int, timepoint: dt.datetime) -> int:
        """Get the number of births in a given time interval.

        Args:
            num_births_initial: Number of births in the initial year of the simulation.
            timepoint: The current timepoint of the simulation.

        Returns:
            The number of births for the given year.

        Examples:

            >>> from leap.birth import Birth
            >>> from leap.utils import get_data_path
            >>> import datetime as dt
            >>> birth = Birth(
            ...     min_timepoint=dt.datetime(2000, 1, 1),
            ...     province="CA",
            ...     projection_scenario="LG"
            ... )
            >>> birth.get_num_newborn(num_births_initial=100, timepoint=dt.datetime(2000, 1, 1))
            100

            >>> birth.get_num_newborn(num_births_initial=100, timepoint=dt.datetime(2001, 1, 1))
            97

        """
        num_new_born = int(
            math.ceil(
                num_births_initial * self.estimate.get_group((timepoint))["N_relative"].iloc[0] # type: ignore
            ) 
        ) 
        return num_new_born
