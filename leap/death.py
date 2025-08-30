from __future__ import annotations
import copy
import pandas as pd
import numpy as np
from leap.utils import get_data_path, check_year, check_province
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.agent import Agent

logger = get_logger(__name__)


class Death:
    """Contains information about the probability of death for an agent in a given year."""
    def __init__(
        self,
        province: str = "CA",
        starting_year: int = 2000,
        life_table: DataFrameGroupBy | None = None
    ):
        if life_table is None:
            self.life_table = self.load_life_table(starting_year, province)

    @property
    def life_table(self) -> DataFrameGroupBy:
        """A grouped data frame grouped by year. Each data frame contains the following columns:
        
        * ``age (int)``: age of person.
        * ``year (int)``: calendar year.
        * ``F (float)``: the probability of death for a female of a given age in a given
          year.
        * ``M (float)``: the probability of death for a male of a given age in a given year.
        """
        return self._life_table
    
    @life_table.setter
    def life_table(self, life_table: DataFrameGroupBy):
        self._life_table = life_table

    def load_life_table(self, starting_year: int, province: str) -> DataFrameGroupBy:
        """Load the life table data.
        
        Args:
            starting_year: The year to start the data at.
            province: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
        
        Returns:
            A grouped data frame grouped by year.
            
            Each data frame contains the following columns:

            * ``age (int)``: age of person.
            * ``year (int)``: calendar year.
            * ``F (float)``: the probability of death for a female of a given age in a given
                year.
            * ``M (float)``: the probability of death for a male of a given age in a given year.
        """
        df = pd.read_csv(
            get_data_path("processed_data/life_table.csv")
        )
        check_year(starting_year, df)
        check_province(province)

        df = df[
            (df["year"] >= starting_year) &
            (df["province"] == province)
        ]
        df.drop(columns=["se", "province"], inplace=True)
        df = df.pivot(index=["age", "year"], columns=["sex"], values="prob_death").reset_index()
        df.columns.name = ""
        grouped_df = df.groupby(["year"])
        return grouped_df

    def agent_dies(self, agent: Agent) -> bool:
        """Determine whether or not the agent dies in a given year, based on age and sex.

        Args:
            agent: A person in the model.

        Returns:
            ``True`` if the agent dies, ``False`` otherwise.
        """

        df = self.life_table.get_group((agent.year,))
        p = df[df["age"] == agent.age][str(agent.sex)].values[0]
        is_dead = bool(np.random.binomial(1, p))
        return is_dead
