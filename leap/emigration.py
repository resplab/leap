from __future__ import annotations
from ast import parse
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from leap.utils import get_data_path, check_timepoint, check_province, check_projection_scenario, \
    get_time_interval_tag
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.utils import Sex

logger = get_logger(__name__)


class Emigration:
    """A class containing information about emigration from Canada."""
    def __init__(
        self,
        min_timepoint: dt.datetime = dt.datetime(2000, 1, 1),
        province: str = "CA",
        population_growth_type: str = "LG",
        table: DataFrameGroupBy | None = None,
        time_interval: dt.timedelta | relativedelta = relativedelta(years=1)
    ):
        if table is None:
            self.table = self.load_emigration_table(
                min_timepoint, province, population_growth_type, time_interval
            )
        else:
            self.table = table

    @property
    def table(self) -> DataFrameGroupBy:
        """Grouped dataframe (by timepoint) giving the probability of emigration for a given age,
        province, sex, and growth scenario:

        * ``timepoint``: timepoint in the range 2001 - 2065.
        * ``age``: integer age.
        * ``M``: the probability of a male emigrating.
        * ``F``: the probability of a female emigrating.

        See ``processed_data/migration/emigration_table.csv``.
        """
        return self._table
    
    @table.setter
    def table(self, table: DataFrameGroupBy):
        self._table = table

    def load_emigration_table(
        self,
        min_timepoint: dt.datetime,
        province: str,
        population_growth_type: str,
        time_interval: dt.timedelta | relativedelta
    ) -> DataFrameGroupBy:
        """Load the data from ``processed_data/migration/emigration_table.csv``.

        Args:
            min_timepoint: the timepoint for the data to start at. Must be between 2001-2065.
            province: a string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".
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
            A dataframe grouped by timepoint, giving the probability of emigration for a given
            age, province, sex, and growth scenario.
        """
        time_interval_tag = get_time_interval_tag(time_interval)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_interval_tag}/migration/emigration_table.csv"),
            parse_dates=["timepoint"]
        )
        check_timepoint(min_timepoint + time_interval, df)
        check_province(province)
        check_projection_scenario(population_growth_type)

        df = df[
            (df["timepoint"] >= min_timepoint) &
            (df["province"] == province) &
            (df["proj_scenario"] == population_growth_type)
        ]
        df.drop(columns=["province", "proj_scenario"], inplace=True)
        grouped_df = df.groupby("timepoint")
        return grouped_df

    def compute_probability(
        self,
        timepoint: dt.datetime,
        age: int,
        sex: str | Sex
    ) -> bool:
        """Determine the probability of emigration of an agent (person) in a given timepoint.

        Args:
            timepoint: The timepoint, e.g. ``2022``.
            age: Age of the person.
            sex: Sex of the person, "M" = male, "F" = female.

        Returns:
            ``True`` if the person emigrates, ``False`` otherwise.

        Examples:

            >>> emigration = Emigration()
            >>> emigration.compute_probability(timepoint=2022, age=0, sex="F")
            False

        """

        if age == 0:
            return False
        else:
            df = self.table.get_group(timepoint)
            p = df[df["age"] == min(age, 100)][str(sex)].values[0]
            return bool(np.random.binomial(1, p))
