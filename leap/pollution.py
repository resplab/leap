from __future__ import annotations
import pathlib
import os
import numpy as np
import pandas as pd
import pygrib
from leap.utils import get_data_path
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
        * ``year``: the year for the pollution data projection.
        * ``month``: the month for the pollution data projection.
        * ``date``: the data for the pollution data projection, first of the month.
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
        year: the year for the pollution data projection.
        month: the month for the pollution data projection.
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
        year: int,
        month: int,
        SSP: str,
        pollution_table: PollutionTable | None = None
    ):
        if pollution_table is None:
            pollution_table = PollutionTable()

        df = pollution_table.data.get_group(SSP)
        df = df[(df["CDUID"] == cduid) & (df["year"] == year) & (df["month"] == month)]
        self.cduid = cduid
        self.year = year
        self.month = month
        self.wildfire_pm25_scaled = df["wildfire_pm25_scaled"].values[0]
        self.total_pm25 = df["total_pm25"].values[0]
        self.SSP = SSP


class GribData:
    """A class containing GRIB data on air pollution.

    Attributes:
        year: year the data was collected.
        month: month the data was collected.
        day: day the data was collected.
        projection: which map projection was used. See `gridType` on the
            `GRIB keys <https://confluence.ecmwf.int/display/ECC/GRIB+Keys>`_ page.
        longitudes: a list of longitude values.
        latitudes: a list of latitude values.
        values: a list of the values of interest at a specified longitude and latitude.
            For example, it could be the PM2.5 concentration.
    """
    def __init__(
        self,
        file_path: str | pathlib.Path | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        projection: str | None = None,
        longitudes: np.ndarray | None = None,
        latitudes: np.ndarray | None = None,
        values: np.ndarray | None = None
    ):
        if file_path is not None:
            year, month, day, projection, df = self.load_file(file_path)
            self.longitudes = df["longitudes"]
            self.latitudes = df["latitudes"]
            self.values = df["mean"]
        else:
            self.longitudes = longitudes
            self.latitudes = latitudes
            self.values = values

        self.projection = projection
        self.year = year
        self.month = month
        self.day = day

    def load_file(self, file_path: pathlib.Path | str):
        """Load a ``*.grib2`` file and amalgamate the data by taking the mean.

        Args:
            file_path: Full file name of ``*.csv`` file to load the data from.
        """
        df = pd.DataFrame()
        with pygrib.open(file_path) as f:
            index = 0
            for record in f:
                longitudes = invert_longitude(record["longitudes"])
                df = add_record_to_df(df, longitudes, record["latitudes"], record["codedValues"], index)
                index += 1
                year = record["year"]
                month = record["month"]
                day = record["day"]
                projection = record["gridType"]
        df = get_data_average(df, year, month, day, projection)
        return year, month, day, projection, df

    def save(self, file_path: pathlib.Path | str):
        """Save the GribData object to a ``*.csv`` file.

        Args:
            file_path: Full file name of ``*.csv`` file to save the data to.
        """
        df = pd.DataFrame({
            "longitudes": self.longitudes,
            "latitudes": self.latitudes,
            "values": self.values
        })
        df.to_csv(file_path)


def add_record_to_df(
    df: pd.DataFrame,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    values: np.ndarray,
    index: int
) -> pd.DataFrame:
    """Add a new column to the grib data frame.

    Args:
        df: A data frame with the following columns:

            * ``longitudes``: a list of longitude values.
            * ``latitudes``: a list of latitude values.
            * ``value_{index}``: a list of the values of interest at a specified longitude and
              latitude. For example, it could be the PM2.5 concentration. Each value column
              corresponds to either a record in the original grib file, or a file in a folder.
        longitudes: An array of longitude values.
        latitudes: An array of latitude values.
        values: An array of the values of interest at a specified longitude and latitude.
            For example, it could be the PM2.5 concentration.
        index: The index of the iteration loop.

    Returns:
        A dataframe with a new column added.
    """
    colname = f"value_{index}"
    colname_lat = "latitudes"
    colname_lon = "longitudes"
    if index == 0:
        df[colname_lon] = longitudes
        df[colname_lat] = latitudes
        df[colname] = values
    else:
        df[colname] = values
    return df


def get_data_average(df: pd.DataFrame) -> pd.DataFrame:
    """Find the mean of all the ``value_*`` columns, and return a ``GribData`` object.

    Args:
        df: A data frame with the following columns:

            * ``longitudes``: a list of longitude values.
            * ``latitudes``: a list of latitude values.
            * ``value_{index}``: a list of the values of interest at a specified longitude and
              latitude. For example, it could be the PM2.5 concentration. Each value column
              corresponds to either a record in the original grib file, or a file in a folder.
    """
    value_columns = df.columns[df.columns.str.startswith("value")]
    df["mean"] = df[value_columns].mean(axis=1)
    return df


def invert_longitude(longitude: float) -> float:
    """Convert between + degrees East of the Prime Meridian and - degrees West of the Prime Meridian.

    Longitude values can be given as either + degrees East of the Prime Meridian, or as
    - degrees West of the Prime Meridian.

    Args:
        longitude: a number in ``[0, 360)`` giving the degrees east or west of the Prime Meridian.

    Returns:
        The converted longitude.
    """
    if longitude > 0:
        return longitude - 360
    elif longitude < 0:
        return 360 + longitude
    else:
        return longitude


def load_grib_files(folder: str, recursive: bool = False) -> GribData:
    """Load multiple ``*.grib2`` files and amalgamate the data by taking the mean.

    Args:
        folder: The folder containing the ``.grib2`` files to open.
        recursive: If ``True``, iterate through all subdirectories and compute an
            aggregate average. If ``False``, only read ``.grib2`` files in the given directory.
    """
    index = 1
    df = pd.DataFrame()
    for item in os.listdir(folder):
        if os.path.isdir(item):
            if recursive:
                logger.info(f"Reading directory {item}")
                grib_data = load_grib_files(item)
                df = add_record_to_df(
                    df, grib_data.longitudes, grib_data.latitudes, grib_data.values, index
                )
                index += 1
        else:
            logger.info(f"Reading file {item}")
            filename = item
            extension = filename[filename.rfind("."):]
            if extension == ".grib2":
                grib_data = GribData(file_path=filename)
                df = add_record_to_df(
                    df, grib_data.longitudes, grib_data.latitudes, grib_data.values, index
                )
                index += 1
    df = get_data_average(df)
    final_grib_data = GribData(
        year=grib_data.year, month=grib_data.month, day=None,
        projection=grib_data.projection, longitudes=df["longitudes"],
        latitudes=df["latitudes"], values=df["mean"]
    )
    return final_grib_data














