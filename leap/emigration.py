from __future__ import annotations
import pandas as pd
import numpy as np
from leap.utils import get_data_path, check_year, check_province, check_projection_scenario
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
        starting_year: int = 2000,
        province: str = "CA",
        population_growth_type: str = "LG",
        table: DataFrameGroupBy | None = None
    ):
        if table is None:
            self.table = self.load_emigration_table(starting_year, province, population_growth_type)
        else:
            self.table = table

    @property
    def table(self) -> DataFrameGroupBy:
        """Grouped dataframe (by year) giving the probability of emigration for a given age,
        province, sex, and growth scenario:

        * ``year``: integer year the range 2001 - 2065.
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
        self, starting_year: int, province: str, population_growth_type: str
    ) -> DataFrameGroupBy:
        """Load the data from ``processed_data/migration/emigration_table.csv``.

        Args:
            starting_year: the year for the data to start at. Must be between 2001-2065.
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
            A dataframe grouped by year, giving the probability of emigration for a given
            age, province, sex, and growth scenario.
        """
        df = pd.read_csv(
            get_data_path("processed_data/migration/emigration_table.csv")
        )
        check_year(starting_year + 1, df)
        check_province(province)
        check_projection_scenario(population_growth_type)

        df = df[
            (df["year"] >= starting_year) &
            (df["province"] == province) &
            (df["proj_scenario"] == population_growth_type)
        ]
        df.drop(columns=["province", "proj_scenario"], inplace=True)
        grouped_df = df.groupby(["year"])
        return grouped_df

    def compute_probability(self, year: int, age: int, sex: str | Sex) -> bool:
        """Determine the probability of emigration of an agent (person) in a given year.

        Args:
            year: The calendar year, e.g. ``2022``.
            age: Age of the person.
            sex: Sex of the person, "M" = male, "F" = female.

        Returns:
            ``True`` if the person emigrates, ``False`` otherwise.

        Examples:

            >>> emigration = Emigration()
            >>> emigration.compute_probability(year=2022, age=0, sex="F")
            False

        """

        if age == 0:
            return False
        else:
            df = self.table.get_group((year,))
            p = df[df["age"] == min(age, 100)][str(sex)].values[0]
            return bool(np.random.binomial(1, p))
