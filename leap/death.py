from __future__ import annotations
import pathlib
import pandas as pd
import numpy as np
from leap.utils import PROCESSED_DATA_PATH
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.agent import Agent

logger = get_logger(__name__)


class Death:
    """Contains information about the probability of death for an agent in a given year."""
    def __init__(
        self,
        config: dict | None = None,
        province: str = "CA",
        starting_year: int = 2000,
        parameters: dict | None = None,
        life_table: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if config is not None:
            self.parameters = config["parameters"]
        elif parameters is not None:
            self.parameters = parameters
        else:
            raise ValueError("Either config dict or parameters must be provided.")

        if life_table is None:
            self.life_table = self.load_life_table(starting_year, province)

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:
            * ``β0``: A number.
            * ``β1``: A number.
            * ``β2``: A number.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["β0", "β1", "β2"]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"Parameter {key} is missing.")
        self._parameters = parameters

    @property
    def life_table(self) -> pd.api.typing.DataFrameGroupBy:
        """A grouped data frame grouped by year. Each data frame contains the following columns:
            * ``age`` (int): age of person.
            * ``year`` (int): calendar year.
            * ``F`` (float): the probability of death for a female of a given age in a given
                year.
            * ``M`` (float): the probability of death for a male of a given age in a given year.
        """
        return self._life_table
    
    @life_table.setter
    def life_table(self, life_table: pd.api.typing.DataFrameGroupBy):
        self._life_table = life_table

    def load_life_table(self, starting_year: int, province: str) -> pd.api.typing.DataFrameGroupBy:
        """Load the life table data.
        
        Args:
            starting_year: the year to start the data at.
            province: a string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".
        
        Returns:
            A grouped data frame grouped by year. Each data frame contains the following columns:
                * ``age`` (int): age of person.
                * ``year`` (int): calendar year.
                * ``F`` (float): the probability of death for a female of a given age in a given
                  year.
                * ``M`` (float): the probability of death for a male of a given age in a given year.
        """
        df = pd.read_csv(
            pathlib.Path(PROCESSED_DATA_PATH, "master_life_table.csv")
        )
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

        is_dead = False
        df = self.life_table.get_group((agent.year,))
        p = df[df["age"] == agent.age][str(agent.sex)].values[0]

        if p == 1:
            is_dead = True
        else:
            # calibration
            odds_ratio = p / (1 - p) * np.exp(
                self.parameters["β0"] +
                self.parameters["β1"] * agent.year_index +
                self.parameters["β2"] * agent.age
            )
            p = max(min(odds_ratio / (1 + odds_ratio), 1), 0)
            is_dead = bool(np.random.binomial(1, p))
        return is_dead
