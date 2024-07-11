import pathlib
import math
import pandas as pd
import numpy as np
from leap.utils import PROCESSED_DATA_PATH
from leap.logger import get_logger

logger = get_logger(__name__)


class Immigration:
    """A class containing information about immigration to Canada.

    Attributes:
        table (pd.api.typing.DataFrameGroupBy): A dataframe grouped by year, giving the probability
            of immigration for a given age, province, sex, and growth scenario:
                * ``year``: integer year the range 2001 - 2065.
                * ``age``: integer age.
                * ``sex``: integer, 0 = female, 1 = male
                * ``prop_immigrants_birth``: the number of immigrants relative to the number of
                  births in that year. To compute the number of immigrants in a given year, multiply
                  the number of births by ``prop_immigrants_birth``.
                * ``prop_immigrants_year``: the proportion of immigrants for a given age and sex
                  relative to the total number of immigrants for a given year and
                  projection scenario.
            See ``master_immigration_table.csv``.
    """
    def __init__(
        self,
        starting_year: int = 2000,
        province: str = "CA",
        population_growth_type: str = "LG",
        max_age: int = 111,
        table: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if table is None:
            self.table = self.load_immigration_table(
                starting_year, province, population_growth_type, max_age
            )
        else:
            self.table = table

    def load_immigration_table(
        self, starting_year: int, province: str, population_growth_type: str, max_age: int
    ):
        """Load the data from ``master_immigration_table.csv``.

        Args:
            starting_year (int): the year for the data to start at. Must be between 2001-2065.
            province (str): a string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".
            population_growth_type: Population growth type, one of:
                ["past", "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", FA", "SA"].
                See `Stats Canada <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.

        Returns:
            pd.api.typing.DataFrameGroupBy: A dataframe grouped by year, giving the probability of
                immigration for a given age, province, sex, and growth scenario.
        """
        df = pd.read_csv(
            pathlib.Path(PROCESSED_DATA_PATH, "migration/master_immigration_table.csv")
        )

        df = df[
            (df["age"] <= max_age) &
            (df["year"] >= starting_year) &
            (df["province"] == province) &
            (df["proj_scenario"] == population_growth_type)
        ]
        df.drop(columns=["province", "proj_scenario"], inplace=True)
        grouped_df = df.groupby(["year"])
        return grouped_df

    def get_num_new_immigrants(self, num_new_born: int, year: int) -> int:
        """Get the number of new immigrants to Canada in a given year.

        Args:
            num_new_born (int): The number of births in the given year of the simulation.
            year (int): The calendar year.

        Returns:
            int: the number of new immigrants to Canada in a given year.
        """

        num_new_immigrants = int(math.ceil(
            num_new_born * np.sum(self.table.get_group((year))["prop_immigrants_birth"])
        ))
        return num_new_immigrants
