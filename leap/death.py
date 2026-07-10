from __future__ import annotations
import copy
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from leap.utils import get_data_path, check_timepoint, check_province, get_time_delta_tag
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
        projection_scenario: str = "M3",
        min_timepoint: dt.datetime = dt.datetime(2000, 1, 1),
        life_table: DataFrameGroupBy | None = None,
        time_delta: dt.timedelta | relativedelta = relativedelta(years=1)
    ):
        if life_table is None:
            self.life_table = self.load_life_table(
                min_timepoint, province, projection_scenario, time_delta
            )

    @property
    def life_table(self) -> DataFrameGroupBy:
        """A grouped data frame grouped by timepoint. Each data frame contains the following columns:
        
        * ``age (int)``: age of person.
        * ``timepoint (dt.datetime)``: timepoint.
        * ``F (float)``: the probability of death for a female of a given age in a given
          timepoint.
        * ``M (float)``: the probability of death for a male of a given age in a given timepoint.
        """
        return self._life_table
    
    @life_table.setter
    def life_table(self, life_table: DataFrameGroupBy):
        self._life_table = life_table

    def load_life_table(
        self,
        min_timepoint: dt.datetime,
        province: str,
        projection_scenario: str,
        time_delta: dt.timedelta | relativedelta
    ) -> DataFrameGroupBy:
        """Load the life table data.
        
        Args:
            min_timepoint: The timepoint to start the data at.
            province: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
            projection_scenario: The projection scenario for the population data. One of:

                * ``LG``: low-growth projection
                * ``HG``: high-growth projection
                * ``M1``: medium-growth 1 projection
                * ``M2``: medium-growth 2 projection
                * ``M3``: medium-growth 3 projection
                * ``M4``: medium-growth 4 projection
                * ``M5``: medium-growth 5 projection
                * ``M6``: medium-growth 6 projection
                * ``FA``: fast-aging projection
                * ``SA``: slow-aging projection

                See: `StatCan Projection Scenarios
                <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
            time_delta: The time interval to use for the life table, e.g. 1 year, 5 years, etc.
        
        Returns:
            A grouped data frame grouped by timepoint.
            
            Each data frame contains the following columns:

            * ``age (int)``: age of person.
            * ``timepoint (dt.datetime)``: timepoint.
            * ``F (float)``: the probability of death for a female of a given age in a given
                timepoint.
            * ``M (float)``: the probability of death for a male of a given age in a given timepoint.
        """
        time_delta_tag = get_time_delta_tag(time_delta)
        check_province(province)

        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/death/life_table_{province}.csv"),
            parse_dates=["timepoint"]
        )
        check_timepoint(min_timepoint, df)

        df = df[
            (df["timepoint"] >= min_timepoint) &
            (df["province"] == province) &
            (df["projection_scenario"].isin(["past", projection_scenario]))
        ]
        df.drop(columns=["province", "projection_scenario"], inplace=True)
        df = df.pivot(index=["age", "timepoint"], columns=["sex"], values="prob_death").reset_index()
        df.columns.name = ""
        grouped_df = df.groupby("timepoint")
        return grouped_df

    def agent_dies(self, agent: Agent) -> bool:
        """Determine whether or not the agent dies in a given timepoint, based on age and sex.

        Args:
            agent: A person in the model.

        Returns:
            ``True`` if the agent dies, ``False`` otherwise.
        """

        df = self.life_table.get_group(agent.timepoint)
        p = df[df["age"] == agent.age][str(agent.sex)].values[0]
        is_dead = bool(np.random.binomial(1, p))
        return is_dead
