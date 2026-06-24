from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from leap.utils import get_data_path, check_timepoint, check_province, check_projection_scenario, \
    get_time_delta_tag, TimeDelta
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.utils import Sex
    from dateutil.relativedelta import relativedelta

logger = get_logger(__name__)


class Emigration:
    """A class containing information about emigration from Canada."""
    def __init__(
        self,
        min_timepoint: dt.datetime = dt.datetime(2000, 1, 1),
        province: str = "CA",
        population_growth_type: str = "LG",
        table: DataFrameGroupBy | None = None,
        time_delta: dt.timedelta | relativedelta | TimeDelta = TimeDelta(years=1)
    ):
        if table is None:
            self.table = self.load_emigration_table(
                min_timepoint, province, population_growth_type, time_delta
            )
        else:
            self.table = table

    @property
    def table(self) -> DataFrameGroupBy:
        """Grouped dataframe (by timepoint) giving the probability of emigration for a given age,
        province, sex, and growth scenario:

        * ``timepoint``: timepoint in the range 2001-2068 (CA) or 2001-2043 (BC).
        * ``age``: integer age.
        * ``sex``: one of ``M`` = male, ``F`` = female.
        * ``prob_emigration``: the per-person probability of emigrating. Zero for cells where
          the net population change was non-negative (i.e. no net emigration).

        See ``processed_data/{time_delta_tag}/migration/migration_table.csv``.
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
        time_delta: dt.timedelta | relativedelta | TimeDelta
    ) -> DataFrameGroupBy:
        """Load the data from ``processed_data/{time_delta_tag}/migration/migration_table.csv``.

        Args:
            min_timepoint: the timepoint for the data to start at. Must be between 2001-2068 (CA)
                or 2001-2043 (BC).
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
        time_delta_tag = get_time_delta_tag(time_delta)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/migration/migration_table.csv"),
            parse_dates=["timepoint"]
        )
        check_province(province)
        check_projection_scenario(population_growth_type)
        check_timepoint(min_timepoint + time_delta, df[df["province"] == province])

        df = df[
            (df["timepoint"] >= min_timepoint) &
            (df["province"] == province) &
            (df["projection_scenario"].isin(["past", population_growth_type]))
        ]

        df.drop(columns=["province", "projection_scenario"], inplace=True)
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
            timepoint: The timepoint, e.g. ``dt.datetime(2022, 1, 1)``.
            age: Age of the person.
            sex: Sex of the person, "M" = male, "F" = female.

        Returns:
            ``True`` if the person emigrates, ``False`` otherwise.

        Examples:

            >>> emigration = Emigration()
            >>> emigration.compute_probability(timepoint=dt.datetime(2022, 1, 1), age=0, sex="F")
            False

        """

        if age == 0:
            return False
        else:
            df = self.table.get_group(timepoint)
            p = df[(df["age"] == min(age, 100)) & (df["sex"] == str(sex))]["prob_emigration"].values[0]
            return bool(np.random.binomial(1, p))
