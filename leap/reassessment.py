from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from leap.utils import get_data_path, check_timepoint, check_province, get_time_delta_tag, TimeDelta
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.agent import Agent
    from dateutil.relativedelta import relativedelta

logger = get_logger(__name__)


class Reassessment:
    """A class containing information about asthma diagnosis reassessment."""
    def __init__(
        self,
        min_timepoint: dt.datetime = dt.datetime(2000, 1, 1),
        province: str = "CA",
        table: DataFrameGroupBy | None = None,
        time_delta: dt.timedelta | relativedelta | TimeDelta = TimeDelta(years=1)
    ):
        if table is None:
            self.table = self.load_reassessment_table(min_timepoint, province, time_delta)
        else:
            self.table = table

    @property
    def table(self) -> DataFrameGroupBy:
        """Grouped dataframe (by timepoint) giving the probability of an agent still having asthma after
        reassessment for a given age, province, and sex:

        * ``timepoint (dt.datetime)``: the timepoint, e.g. ``2024-01-01``.
        * ``age (int)``: integer age.
        * ``sex (str)``: one of ``"M"`` = male, ``"F"`` = female.
        * ``prob (float)``: the probability that an agent previously diagnosed with asthma
          maintains their asthma diagnosis after reassessment in the given timepoint.
          Value in ``[0, 1]``.
        * ``province (str)``: a string indicating the province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.

        See ``processed_data/asthma_reassessment.csv``.
        """
        return self._table
    
    @table.setter
    def table(self, table: DataFrameGroupBy):
        self._table = table

    def load_reassessment_table(
        self,
        min_timepoint: dt.datetime,
        province: str,
        time_delta: dt.timedelta | relativedelta | TimeDelta
    ) -> DataFrameGroupBy:
        """Load the asthma diagnosis reassessment table.

        Args:
            min_timepoint: the timepoint to start the data at.
            province: a string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".
            time_delta: The time interval to use for the reassessment table, e.g. 1 year,
                5 years, etc.

        Returns:
            A grouped data frame grouped by timepoint.
            Each data frame contains the following columns:

            * ``timepoint (dt.datetime)``: the timepoint, e.g. ``2024-01-01``.
            * ``age (int)``: integer age.
            * ``sex (str)``: one of ``"M"`` = male, ``"F"`` = female.
            * ``prob (float)``: the probability that an agent previously diagnosed with asthma
              maintains their asthma diagnosis after reassessment in the given timepoint.
              Value in ``[0, 1]``.
            * ``province (str)``: a string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
        """
        time_delta_tag = get_time_delta_tag(time_delta)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/asthma_reassessment.csv"),
            parse_dates=["timepoint"]
        )
        check_timepoint(min_timepoint, df)
        check_province(province)

        df = df[
            (df["timepoint"] >= min_timepoint) &
            (df["province"] == province)
        ]
        grouped_df = df.groupby("timepoint")
        return grouped_df

    def agent_has_asthma(self, agent: Agent) -> bool:
        """If an agent has been diagnosed with asthma, determine whether the agent still has asthma.

        Asthma is not curable, but it can be mistaken for other respiratory diseases, and so a
        person could be diagnosed with asthma and later find this was a misdiagnosis. Additionally,
        asthma can go into a period of dormancy, during which time the person does not have any
        asthma symptoms.

        Args:
            agent: A person in the model.

        Returns:
            Whether the person still has asthma after reassessing the diagnosis.

        Examples:

            >>> from leap.agent import Agent
            >>> from leap.reassessment import Reassessment
            >>> import datetime as dt
            >>> reassessment = Reassessment()
            >>> agent = Agent(
            ...     age=3,
            ...     timepoint=dt.datetime(2024, 1, 1),
            ...     timepoint_index=0,
            ...     sex="F",
            ...     has_asthma=False,
            ...     num_antibiotic_use=0,
            ...     has_family_history=False
            ... )
            >>> has_asthma = reassessment.agent_has_asthma(agent)
            >>> print(
            ...     f"Agent was not diagnosed with asthma previously, "
            ...     f"agent reassessed to have asthma?: {has_asthma}"
            ... )
            Agent was not diagnosed with asthma previously, agent reassessed to have asthma?: False

        """
        max_year = max(np.unique([key for key in self.table.groups.keys()]))
        if agent.age < 4:
            return agent.has_asthma
        else:
            df = self.table.get_group(min(agent.timepoint, max_year))
            df = df.loc[
                (df["age"] == agent.age) &
                (df["sex"] == str(agent.sex))
            ]
            probability = df["prob"].values[0]
            return bool(np.random.binomial(1, probability))
