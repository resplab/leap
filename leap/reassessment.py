from __future__ import annotations
import pandas as pd
import numpy as np
from leap.utils import get_data_path
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.agent import Agent

logger = get_logger(__name__)


class Reassessment:
    """A class containing information about asthma diagnosis reassessment.

    Attributes:
        table (pd.api.typing.DataFrameGroupBy): A grouped data frame grouped by year.
            Each data frame contains the following columns:
                * ``year`` (int): calendar year.
                * ``age`` (int): age of person.
                * ``F`` (float): the probability that a female agent still has asthma.
                * ``M`` (float): the probability that a male agent still has asthma.
                * ``province`` (str): a string indicating the province abbreviation, e.g. "BC".
                    For all of Canada, set province to "CA".
            See ``master_asthma_reassessment.csv``.
    """
    def __init__(
        self,
        starting_year: int = 2000,
        province: str = "CA",
        table: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if table is None:
            self.table = self.load_reassessment_table(starting_year, province)
        else:
            self.table = table

    def load_reassessment_table(
        self, starting_year: int, province: str
    ) -> pd.api.typing.DataFrameGroupBy:
        """Load the asthma diagnosis reassessment table.

        Args:
            starting_year (int): the year to start the data at.
            province (str): a string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".

        Returns:
            pd.api.typing.DataFrameGroupBy: A grouped data frame grouped by year.
                Each data frame contains the following columns:
                    * ``year`` (int): calendar year.
                    * ``age`` (int): age of person.
                    * ``F`` (float): the probability that a female agent still has asthma.
                    * ``M`` (float): the probability that a male agent still has asthma.
                    * ``province`` (str): a string indicating the province abbreviation, e.g. "BC".
                        For all of Canada, set province to "CA".
        """
        df = pd.read_csv(
            get_data_path("processed_data", "master_asthma_reassessment.csv")
        )
        df = df[
            (df["year"] >= starting_year) &
            (df["province"] == province)
        ]
        grouped_df = df.groupby(["year"])
        return grouped_df

    def agent_has_asthma(self, agent: Agent) -> bool:
        """If an agent has been diagnosed with asthma, determine whether the agent still has asthma.

        Asthma is not curable, but it can be mistaken for other respiratory diseases, and so a
        person could be diagnosed with asthma and later find this was a misdiagnosis. Additionally,
        asthma can go into a period of dormancy, during which time the person does not have any
        asthma symptoms.

        Args:
            agent (Agent): Agent module, see `Agent`.

        """
        max_year = max(np.unique([key for key in self.table.groups.keys()]))
        if agent.age < 4:
            return agent.has_asthma
        else:
            df = self.table.get_group(min(agent.year, max_year))
            df = df[df["age"] == agent.age]
            sex = "F" if agent.sex == 0 else "M"
            probability = df[sex].values[0]
            return bool(np.random.binomial(1, probability))
