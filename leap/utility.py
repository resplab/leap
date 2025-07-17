from __future__ import annotations
import pandas as pd
import numpy as np
from leap.utils import get_data_path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
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

        table: A grouped data frame grouped by age and sex, containing information about
            EuroQol Group's quality of life metric called the EQ-5D.
            Each data frame contains the following columns:

            * ``age (int)``: integer age, range ``[0, 111]``.
            * ``sex (str)``: sex of a person, either "M" or "F".
            * ``eq5d (float)``: the quality of life.
            * ``sd (float)``: standard deviation of the ``eq5d`` value.

            See ``processed_data/eq5d_canada.csv``.
    """
    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        table: DataFrameGroupBy | None = None
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

    def load_eq5d(self) -> DataFrameGroupBy:
        df = pd.read_csv(get_data_path("processed_data/eq5d_canada.csv"))
        grouped_df = df.groupby(["age", "sex"])
        return grouped_df

    def compute_utility(self, agent: Agent) -> float:
        """Compute the utility for the current year due to asthma exacerbations and control.

        If the agent (person) doesn't have asthma, return the baseline utility.

        Args:
            agent: A person in the model.
        """
        baseline = float(self.table.get_group((agent.age, str(agent.sex)))["eq5d"].iloc[0])
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
