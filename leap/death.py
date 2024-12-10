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
    """A class containing information about the probability of death for an agent in a given year.

    Attributes:
        parameters (dict): A dictionary containing the following keys:
            * ``β0``: A number.
            * ``β1``: A number.
            * ``β2``: A number.
        life_table (pd.api.typing.DataFrameGroupBy): TODO.
    """
    def __init__(
        self,
        config: dict | None = None,
        province: str = "CA",
        starting_year: int = 2000,
        parameters: dict | None = None,
        life_table: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if config is None and parameters is None:
            raise ValueError("Either config dict or parameters must be provided.")
        elif config is not None:
            self.parameters = config["parameters"]
        else:
            self.parameters = parameters

        if life_table is None:
            self.life_table = self.load_life_table(starting_year, province)

    def load_life_table(self, starting_year: int, province: str) -> pd.DataFrame:
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
            agent (Agent): A person in the model.
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
            is_dead = np.random.binomial(1, p)
        return is_dead
