import pandas as pd
import numpy as np
import pathlib
from leap.utils import PROCESSED_DATA_PATH
from leap.logger import get_logger

logger = get_logger(__name__)


class Emigration:
    """A class containing information about emigration from Canada.

    Attributes:
        table (pd.api.typing.DataFrameGroupBy): A dataframe grouped by year, giving the probability of
            emigration for a given age, province, sex, and growth scenario:
                * ``year``: integer year the range 2001 - 2065.
                * ``age``: integer age.
                * ``M``: the probability of a male emigrating.
                * ``F``: the probability of a female emigrating.
            See ``master_emigration_table.csv``.
    """
    def __init__(
        self,
        starting_year: int = 2000,
        province: str = "CA",
        population_growth_type: str = "LG",
        table: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if table is None:
            self.table = self.load_emigration_table(starting_year, province, population_growth_type)
        else:
            self.table = table

    def load_emigration_table(self, starting_year: int, province: str, population_growth_type: str):
        """Load the data from ``master_emigration_table.csv``.

        Args:
            starting_year (int): the year for the data to start at. Must be between 2001-2065.
            province (str): a string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".
            population_growth_type: Population growth type, one of:
                ["past", "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", FA", "SA"].
                See `Stats Canada <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.

        Returns:
            pd.api.typing.DataFrameGroupBy: A dataframe grouped by year, giving the probability of
                emigration for a given age, province, sex, and growth scenario.
        """
        df = pd.read_csv(
            pathlib.Path(PROCESSED_DATA_PATH, "migration/master_emigration_table.csv")
        )
        df = df[
            (df["year"] >= starting_year) &
            (df["province"] == province) &
            (df["proj_scenario"] == population_growth_type)
        ]
        df.drop(columns=["province", "proj_scenario"], inplace=True)
        grouped_df = df.groupby(["year"])
        return grouped_df

    def compute_probability(self, year: int, age: int, sex: bool) -> bool:
        """Determine the probability of emigration of an agent (person) in a given year.

        Args:
            year (int): The calendar year.
            age (int): Age of the person.
            sex (bool): Sex of the person, 1 = male, 0 = female.
        """

        if age == 0:
            return False
        else:
            sex = "M" if sex == 1 else "F"
            df = self.table.get_group((year))
            p = df[df["age"] == min(age, 100)][sex].values[0]
            return bool(np.random.binomial(1, p))
