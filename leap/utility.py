from __future__ import annotations
import pandas as pd
import numpy as np
import pathlib
from leap.utils import PROCESSED_DATA_PATH
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.agent import Agent


class Utility:
    """A class containing information about the disutility from having asthma.

    Attributes:
        parameters: A dictionary containing the following keys:
            * ``βcontrol``: A vector of 3 parameters to be multiplied by the control levels, i.e.

              .. code-block:: python

                βcontrol1 * fully_controlled +
                βcontrol2 * partially_controlled +
                βcontrol3 * uncontrolled

            * ``βexac_sev_hist``: A vector of 4 parameters to be multiplied by the exacerbation
              severity history, i.e.

              .. code-block:: python
              
                βexac_sev_hist1 * mild + βexac_sev_hist2 * moderate +
                βexac_sev_hist3 * severe + βexac_sev_hist4 * very_severe

        table: A grouped data frame grouped by age and sex,
            containing information about EuroQol Group's quality of life metric called the EQ-5D.
            Each data frame contains the following columns:
                * ``age``: integer age.
                * ``sex``: sex of a person, 1 = male, 0 = female
                * ``eq5d``: float, the quality of life.
                * ``se``: float, standard error.
            See ``eq5d_canada.csv``.
    """
    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        table: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if config is not None:
            self.parameters = config["parameters"]
        elif parameters is not None:
            self.parameters = parameters
        else:
            raise ValueError("Either config dict or parameters must be provided.")

        if table is None:
            self.table = self.load_eq5d()
        else:
            self.table = table

        self.parameters["βexac_sev_hist"] = np.array(self.parameters["βexac_sev_hist"])
        self.parameters["βcontrol"] = np.array(self.parameters["βcontrol"])

    def load_eq5d(self) -> pd.api.typing.DataFrameGroupBy:
        df = pd.read_csv(pathlib.Path(PROCESSED_DATA_PATH, "eq5d_canada.csv"))
        grouped_df = df.groupby(["age", "sex"])
        return grouped_df

    def compute_utility(self, agent: Agent) -> float:
        """Compute the utility for the current year due to asthma exacerbations and control.

        If the agent (person) doesn't have asthma, return the baseline utility.

        Args:
            agent: A person in the model.
        """
        baseline = float(self.table.get_group((agent.age, int(agent.sex)))["eq5d"].iloc[0])
        if not agent.has_asthma:
            return baseline
        else:
            disutility_exac = np.sum(
                agent.exacerbation_severity_history.current_year * self.parameters["βexac_sev_hist"]
            )
            disutility_control = np.sum(
                agent.control_levels.as_array() * self.parameters["βcontrol"]
            )
            return max(0, (baseline - disutility_exac - disutility_control))
