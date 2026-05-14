from __future__ import annotations
import math
import pandas as pd
import numpy as np
import datetime as dt
from leap.utils import get_data_path, check_province, check_timepoint, check_projection_scenario, \
    get_time_delta_tag, TimeDelta
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from dateutil.relativedelta import relativedelta

logger = get_logger(__name__)


class Immigration:
    """A class containing information about immigration to Canada."""
    def __init__(
        self,
        min_timepoint: dt.datetime = dt.datetime(2000, 1, 1),
        province: str = "CA",
        population_growth_type: str = "LG",
        max_age: int = 111,
        table: DataFrameGroupBy | None = None,
        time_delta: relativedelta | dt.timedelta | TimeDelta = TimeDelta(years=1)
    ):
        if table is None:
            self.table = self.load_immigration_table(
                min_timepoint, province, population_growth_type, max_age, time_delta
            )
        else:
            self.table = table

    @property
    def table(self) -> DataFrameGroupBy:
        """Grouped dataframe (by timepoint) giving the probability of immigration for a given age,
        province, sex, and growth scenario:

        * ``timepoint``: integer timepoint the range ``2001 - 2065``.
        * ``age``: integer age.
        * ``sex``: integer, ``0 = female``, ``1 = male``.
        * ``prop_immigrants_birth``: The number of immigrants relative to the number of
          births in that timepoint. To compute the number of immigrants in a given timepoint, multiply
          the number of births by ``prop_immigrants_birth``.
        * ``prop_immigrants_timepoint``: The proportion of immigrants for a given age and sex
          relative to the total number of immigrants for a given timepoint and projection scenario.

        See ``processed_data/migration/immigration_table.csv``.
        """
        return self._table
    
    @table.setter
    def table(self, table: DataFrameGroupBy):
        self._table = table

    def load_immigration_table(
        self,
        min_timepoint: dt.datetime,
        province: str,
        population_growth_type: str,
        max_age: int,
        time_delta: dt.timedelta | relativedelta | TimeDelta
    ) -> DataFrameGroupBy:
        """Load the data from ``processed_data/migration/immigration_table.csv``.

        Args:
            min_timepoint: The timepoint for the data to start at. Must be between ``2001-2065``.
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
            max_age: The maximum age to include in the data. Must be between 0 and 111, inclusive.
            time_delta: The time interval to use for the data.

        Returns:
            A dataframe grouped by timepoint, giving the probability of immigration for a given age,
            province, sex, and growth scenario.
        """
        time_delta_tag = get_time_delta_tag(time_delta)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/migration/migration_table.csv"),
            parse_dates=["timepoint"]
        )
        check_timepoint(min_timepoint + time_delta, df)
        check_province(province)
        check_projection_scenario(population_growth_type)

        df = df[
            (df["delta_n"] > 0) &
            (df["age"] <= max_age) &
            (df["timepoint"] >= min_timepoint) &
            (df["province"] == province) &
            (df["projection_scenario"] == population_growth_type)
        ]
        df = df.drop(columns=["province", "projection_scenario", "delta_n", "prop_emigrants_year", "prob_emigration"])
        df = df.rename(columns={"prop_migrants_birth": "prop_immigrants_birth"})
        df["sex"] = df["sex"].replace({"F": 0, "M": 1})
        for timepoint in df["timepoint"].unique():
            prop_immigrants_timepoint = df.loc[df["timepoint"] == timepoint]["prop_immigrants_timepoint"].copy()
            sum_timepoint = prop_immigrants_timepoint.sum()
            df["prop_immigrants_timepoint"] = df.apply(
                lambda x: x["prop_immigrants_timepoint"] / sum_timepoint
                    if x["timepoint"] == timepoint else x["prop_immigrants_timepoint"],
                axis=1
            )
        grouped_df = df.groupby("timepoint")
        return grouped_df

    def get_num_new_immigrants(self, num_new_born: int, timepoint: dt.datetime) -> int:
        """Get the number of new immigrants to Canada in a given timepoint.

        Args:
            num_new_born: The number of births in the given timepoint of the simulation.
            timepoint: The timepoint as a datetime object.

        Returns:
            The number of new immigrants to Canada in a given time interval.

        Examples:

            >>> from leap.immigration import Immigration
            >>> import datetime as dt
            >>> immigration = Immigration(
            ...     min_timepoint=dt.datetime(2000, 1, 1), province="BC", population_growth_type="LG"
            ... )
            >>> n_immigrants = immigration.get_num_new_immigrants(num_new_born=1000, timepoint=dt.datetime(2022, 1, 1))
            >>> print(f"Number of immigrants to BC in 2022 for low growth scenario: {n_immigrants}")
            Number of immigrants to BC in 2022 for low growth scenario: 974

        """

        num_new_immigrants = int(math.ceil(
            num_new_born * np.sum(self.table.get_group(timepoint)["prop_immigrants_birth"])
        ))
        return num_new_immigrants
