from __future__ import annotations
import pathlib
import copy
import pandas as pd
import datetime as dt
from leap.utils import get_data_path, check_timepoint, check_cduid
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy

logger = get_logger(__name__)


class PollutionTable:
    """A class containing information about PM 2.5 pollution in Canada."""
    def __init__(self, data: DataFrameGroupBy | None = None):
        if data is None:
            data = self.load_pollution_data()
        self.data = data

    @property
    def data(self) -> DataFrameGroupBy:
        """A data frame grouped by the SSP scenario, with the following columns:

        * ``CDUID``: the census division identifier.
        * ``timepoint``: the date for the pollution data projection.
        * ``background_pm25``: the average background PM2.5 levels for a given month.
        * ``wildfire_pm25``: the average PM2.5 levels due to wildfires for a given month.
        * ``factor``: the future climate scaling factor, based on the SSP scenario.
        * ``wildfire_pm25_scaled``: ``wildfire_pm25`` * ``factor``.
        * ``total_pm25``: the total average PM2.5 levels for a given month:
            ``wildfire_pm25_scaled + background_pm25``
        * ``SSP``: The Shared Socioeconomic Pathway (SSP) scenario from the IPCC, one of:
            
          - ``SSP1_2.6``: Sustainability - Taking the Green Road, low GHG emissions
          - ``SSP2_4.5``: Middle of the Road, medium GHG emissions
          - ``SSP3_7.0``: Regional Rivalry - A Rocky Road, high GHG emissions
          - ``SSP5_8.5``: Fossil-Driven Development, very high GHG emissions
        """
        return self._data
    
    @data.setter
    def data(self, data: DataFrameGroupBy):
        self._data = data

    def __copy__(self):
        data = self.data.obj.copy()
        return PollutionTable(data=data.groupby("SSP"))

    def __deepcopy__(self, memo):
        data = self.data.obj.copy(deep=True)
        return PollutionTable(data=data.groupby("SSP"))

    def copy(self, deep: bool = True) -> PollutionTable:
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def load_pollution_data(
        self, pm25_data_path: pathlib.Path = get_data_path("processed_data/pollution")
    ) -> DataFrameGroupBy:
        """Load the data from the PM2.5 SSP ``*.csv`` files.

        Args:
            pm25_data_path: Full directory path for the PM2.5 ``*.csv`` files.

        Returns:
            A data frame grouped by the SSP scenario.
        """
        files = pm25_data_path.glob("*.csv")
        pollution_data = pd.DataFrame()
        for file in files:
            df = pd.read_csv(file)
            pollution_data = pd.concat([pollution_data, df])
        grouped_df = pollution_data.groupby("SSP")
        return grouped_df


class Pollution:
    """Contains information about PM2.5 pollution for a census division, date, and SSP scenario.

    Attributes:
        cduid: the census division identifier.
        timepoint: the date for the pollution data projection.
        wildfire_pm25_scaled (float): ``wildfire_pm25`` * ``factor``.
        total_pm25 (float): the total average PM2.5 levels for a given month:
            ``wildfire_pm25_scaled`` + ``background_pm25``.
        SSP: The Shared Socioeconomic Pathway (SSP) scenario from the IPCC, one of:
            
            - ``SSP1_2.6``: Sustainability - Taking the Green Road, low GHG emissions
            - ``SSP2_4.5``: Middle of the Road, medium GHG emissions
            - ``SSP3_7.0``: Regional Rivalry - A Rocky Road, high GHG emissions
            - ``SSP5_8.5``: Fossil-Driven Development, very high GHG emissions
    """

    def __init__(
        self,
        cduid: int,
        timepoint: dt.datetime,
        SSP: str,
        pollution_table: PollutionTable | None = None
    ):
        if pollution_table is None:
            pollution_table = PollutionTable()

        df = pollution_table.data.get_group(SSP)
        check_timepoint(timepoint, df)
        check_cduid(cduid, df)

        df = df[(df["CDUID"] == cduid) & (df["timepoint"] == timepoint)]
        self.cduid = cduid
        self.timepoint = timepoint
        self.wildfire_pm25_scaled = df["wildfire_pm25_scaled"].values[0]
        self.total_pm25 = df["total_pm25"].values[0]
        self.SSP = SSP



