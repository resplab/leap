from __future__ import annotations
import copy
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

            * ``βcontrol``: A vector of 3 parameters giving the disutility from asthma control levels:

              .. code-block:: python

                disutility = \
                    βcontrol[0] * fully_controlled + \
                    βcontrol[1] * partially_controlled + \
                    βcontrol[2] * uncontrolled

            * ``βexac_sev_hist``: A vector of 4 parameters giving the disutility for an asthma
              exacerbation of different severity levels:

              .. code-block:: python
          
                disutility = \
                    βexac_sev_hist[0] * n_mild + βexac_sev_hist[1] * n_moderate + \
                    βexac_sev_hist[2] * n_severe + βexac_sev_hist[3] * n_very_severe

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

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:

        * ``βcontrol``: A vector of 3 parameters giving the disutility from asthma control levels:

          .. code-block:: python

            disutility = \
                βcontrol[0] * fully_controlled + \
                βcontrol[1] * partially_controlled + \
                βcontrol[2] * uncontrolled

        * ``βexac_sev_hist``: A vector of 4 parameters giving the disutility for an asthma
          exacerbation of different severity levels:

          .. code-block:: python
          
            disutility = \
                βexac_sev_hist[0] * n_mild + βexac_sev_hist[1] * n_moderate + \
                βexac_sev_hist[2] * n_severe + βexac_sev_hist[3] * n_very_severe
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["βcontrol", "βexac_sev_hist"]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"The key '{key}' is missing in the parameters.")
        if len(parameters["βcontrol"]) != 3:
            raise ValueError("The length of the 'βcontrol' vector must be 3.")
        if len(parameters["βexac_sev_hist"]) != 4:
            raise ValueError("The length of the 'βexac_sev_hist' vector must be 4.")
        self._parameters = copy.deepcopy(parameters)

    def load_eq5d(self) -> DataFrameGroupBy:
        df = pd.read_csv(get_data_path("processed_data/eq5d_canada.csv"))
        grouped_df = df.groupby(["age", "sex"])
        return grouped_df

    def compute_utility(self, agent: Agent) -> float:
        r"""Compute the utility for the current year due to asthma exacerbations and control.

        If the agent (person) doesn't have asthma, return the baseline utility.
        Otherwise, return the utility:

        .. math::

            u = u_{\text{baseline}} - 
                \sum_{S=1}^{4} d_E(S) \cdot n_E(S) - 
                \sum_{L=1}^{3} d_C(L) \cdot C(L)

        Args:
            agent: A person in the model.
        """
        baseline = float(self.table.get_group((agent.age, str(agent.sex)))["eq5d"].iloc[0])
        if not agent.has_asthma:
            return baseline
        else:
            disutility_exac = np.dot(
                agent.exacerbation_severity_history.current_year, self.parameters["βexac_sev_hist"]
            )
            disutility_control = np.dot(
                agent.control_levels.as_array(), self.parameters["βcontrol"]
            )
            return max(0, (baseline - disutility_exac - disutility_control))
