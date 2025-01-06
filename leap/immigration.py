from __future__ import annotations
import math
import pandas as pd
import numpy as np
from leap.utils import get_data_path, check_province, check_year, check_projection_scenario
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy

logger = get_logger(__name__)


class Immigration:
    """A class containing information about immigration to Canada."""
    def __init__(
        self,
        starting_year: int = 2000,
        province: str = "CA",
        population_growth_type: str = "LG",
        max_age: int = 111,
        table: DataFrameGroupBy | None = None
    ):
        if table is None:
            self.table = self.load_immigration_table(
                starting_year, province, population_growth_type, max_age
            )
        else:
            self.table = table

    @property
    def table(self) -> DataFrameGroupBy:
        """Grouped dataframe (by year) giving the probability of immigration for a given age,
        province, sex, and growth scenario:

        * ``year``: integer year the range ``2001 - 2065``.
        * ``age``: integer age.
        * ``sex``: integer, ``0 = female``, ``1 = male``.
        * ``prop_immigrants_birth``: The number of immigrants relative to the number of
          births in that year. To compute the number of immigrants in a given year, multiply
          the number of births by ``prop_immigrants_birth``.
        * ``prop_immigrants_year``: The proportion of immigrants for a given age and sex
          relative to the total number of immigrants for a given year and projection scenario.

        See ``master_immigration_table.csv``.
        """
        return self._table
    
    @table.setter
    def table(self, table: DataFrameGroupBy):
        self._table = table

    def load_immigration_table(
        self, starting_year: int, province: str, population_growth_type: str, max_age: int
    ) -> DataFrameGroupBy:
        """Load the data from ``master_immigration_table.csv``.

        Args:
            starting_year: The year for the data to start at. Must be between ``2001-2065``.
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
            A dataframe grouped by year, giving the probability of immigration for a given age,
            province, sex, and growth scenario.
        """
        df = pd.read_csv(
            get_data_path("processed_data/migration/master_immigration_table.csv")
        )
        check_year(starting_year, df)
        check_province(province)
        check_projection_scenario(population_growth_type)

        df = df[
            (df["age"] <= max_age) &
            (df["year"] >= starting_year) &
            (df["province"] == province) &
            (df["projection_scenario"] == population_growth_type)
        ]
        df.drop(columns=["province", "projection_scenario"], inplace=True)
        df["sex"].replace({"F": 0, "M": 1}, inplace=True)
        for year in df["year"].unique():
            prop_immigrants_year = df.loc[df["year"] == year]["prop_immigrants_year"].copy()
            sum_year = prop_immigrants_year.sum()
            df["prop_immigrants_year"] = df.apply(
                lambda x: x["prop_immigrants_year"] / sum_year
                    if x["year"] == year else x["prop_immigrants_year"],
                axis=1
            )
        grouped_df = df.groupby(["year"])
        return grouped_df

    def get_num_new_immigrants(self, num_new_born: int, year: int) -> int:
        """Get the number of new immigrants to Canada in a given year.

        Args:
            num_new_born: The number of births in the given year of the simulation.
            year: The calendar year.

        Returns:
            The number of new immigrants to Canada in a given year.

        Examples:

            >>> from leap.immigration import Immigration
            >>> immigration = Immigration(
            ...     starting_year=2000, province="BC", population_growth_type="LG"
            ... )
            >>> n_immigrants = immigration.get_num_new_immigrants(num_new_born=1000, year=2022)
            >>> print(f"Number of immigrants to BC in 2022 for low growth scenario: {n_immigrants}")
            Number of immigrants to BC in 2022 for low growth scenario: 974

        """

        num_new_immigrants = int(math.ceil(
            num_new_born * np.sum(self.table.get_group((year,))["prop_immigrants_birth"])
        ))
        return num_new_immigrants
