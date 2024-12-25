import json
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from leap.utils import get_data_path
from leap.logger import get_logger

logger = get_logger(__name__)


MIN_CENSUS_YEAR = 2021


class CensusTable:
    """A class containing information about Canadian census divisions.

    Please see:
    `Statistics Canada: Table 98-10-0007-01, Population and dwelling counts: Canada and census divisions
    <https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=9810000701>`_.

    """
    def __init__(self, config: dict | None = None, year: int = MIN_CENSUS_YEAR):
        if config is not None:
            self.year = config["year"]
        else:
            self.year = year
        self.data = self.load_census_data()
        self.grouped_data = self.data.groupby(["province"])

    @property
    def data(self) -> pd.DataFrame:
        """A grouped data frame grouped by province containing census population data.
        
        Each data frame contains the following columns:
        
        * ``year (int)``: the year the census data was collected.
        * ``census_division_name (str)``: the federal census division name.
        * ``CDUID (int)``: the census division identifier.
        * ``DGUID (str)``: the dissemination geography unique identifier, in the format:

          .. code-block::

            2021 | A | 0003 | CDUID
            2021 - year the data was collected
            A - administrative (not important, a StatsCan identifier)
            0003 - schema indicating census division
            CDUID - the census division identifier
                
        * ``geographic_area_type (str)``: the census division region type.
        * ``province (str)``: the two-letter province identifier.
        * ``population (int)``: the number of residents living in the census division.
        * ``area_km2 (float)``: the area of the census division in kilometres squared.
        * ``population_density_per_square_km (float)``: the population density per square kilometre.
        """
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame):
        self._data = data

    @property
    def year(self) -> int:
        """The year the census population data was collected."""
        return self._year

    @year.setter
    def year(self, year: int):
        if year < MIN_CENSUS_YEAR:
            raise ValueError(
                f"year must be > {MIN_CENSUS_YEAR}, received {year}."
            )
        self._year = year

    def load_census_data(self) -> pd.DataFrame:
        """Load the census data."""

        df = pd.read_csv(
            get_data_path("processed_data.census_divisions", "master_census_data_2021.csv")
        )
        return df


class CensusDivision:
    """A class containing information about a Canadian census division.

    Please see:
    `Statistics Canada: Table 98-10-0007-01, Population and dwelling counts: Canada and census divisions
    <https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=9810000701>`_.

    If the user provides only the province and year, a census division will be randomly assigned
    based on population.
    """
    def __init__(
        self,
        cduid: int | None = None,
        name: str | None = None,
        year: int = MIN_CENSUS_YEAR,
        province: str = "CA",
        census_table: CensusTable | None = None
    ):
        if cduid is None or name is None:
            if census_table is None:
                census_table = CensusTable(year=year)
            if province == "CA":
                df = census_table.data
            else:
                df = census_table.grouped_data.get_group((province,))

            probabilities = df["population"] / df["population"].sum()
            census_division_id = int(np.random.choice(
                a=df["CDUID"],
                p=probabilities,
                size=1
            )[0])
            census_division_name = df[df["CDUID"] == census_division_id]["census_division_name"].values[0]
            self.name = census_division_name
            self.cduid = census_division_id
            self.year = year
        else:
            self.cduid = cduid
            self.name = name
            self.year = year

    @property
    def cduid(self) -> int:
        """The census division identifier."""
        return self._cduid
    
    @cduid.setter
    def cduid(self, cduid: int):
        self._cduid = cduid

    @property
    def name(self) -> str:
        """The census division name."""
        return self._name
    
    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def year(self) -> int:
        """The year the census population data was collected."""
        return self._year
    
    @year.setter
    def year(self, year: int):
        if year < MIN_CENSUS_YEAR:
            raise ValueError(
                f"year must be > {MIN_CENSUS_YEAR}, received {year}."
            )
        self._year = year


class CensusBoundaries:
    """A class containing information about Canadian census division boundaries.

    Please see:
    `Statistics Canada Census Boundaries
    <https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21>`_:
    
    .. code-block::

        Statistics Canada. 2021 Census – Boundary file.
        Type: Cartographic Boundary File (CBF)
        Administrative Boundaries: Census divisions
        Downloadable: Shapefile (.shp)

    The shapefile given is in the Lambert Conformal Conic projection. For more details, see:
    `Boundary Files, Reference Guide
    <https://www150.statcan.gc.ca/n1/en/pub/92-160-g/92-160-g2021002-eng.pdf?st=0cwqeEP1>`_.

    Attributes:
        shapefile_data: A grouped data frame grouped by province.
            Each data frame contains the following columns:

            * ``geometry``: the ``Shapefile.Polygon`` of the census boundary.
            * ``CDUID``: the census division identifier.
            * ``DGUID``: the dissemination geography unique identifier, in the format:
                .. code-block::

                    2021 | A | 0003 | CDUID
                    2021 - year the data was collected
                    A - administrative (not important, a StatsCan identifier)
                    0003 - schema indicating census division
                    CDUID - the census division identifier

            * ``CDNAME``: the census division name.
            * ``CDTYPE``: the census division type.
            * ``LANDAREA``: the area of the census division in square kilometres.
            * ``PRUID``: integer, province id.
        year: the year the census population data was collected.
        reference_longitude: the reference longitude.
        reference_latitude: the reference latitude.
        first_standard_parallel: the first standard parallel in degrees.
        second_standard_parallel: the second standard parallel in degrees.
        false_easting: false easting.
        false_northing: false northing.
    """
    def __init__(
        self,
        shapefile_data: gpd.GeoDataFrame | None = None,
        shapefile_path: str | None = None,
        metadata_path: str | None = None,
        year: int = MIN_CENSUS_YEAR,
        reference_longitude: float | None = None,
        reference_latitude: float | None = None,
        first_standard_parallel: float | None = None,
        second_standard_parallel: float | None = None,
        false_easting: float | None = None,
        false_northing: float | None = None
    ):
        if metadata_path is not None:
            with open(metadata_path) as file:
                metadata = json.load(file)
            self.year = metadata["year"]
            self.reference_longitude = metadata["reference_longitude"]
            self.reference_latitude = metadata["reference_latitude"]
            self.first_standard_parallel = metadata["first_standard_parallel"]
            self.second_standard_parallel = metadata["second_standard_parallel"]
            self.false_easting = metadata["false_easting"]
            self.false_northing = metadata["false_northing"]
        else:
            if reference_longitude is None:
                raise ValueError("reference_longitude must be provided.")
            else:
                self.reference_longitude = reference_longitude
            if reference_latitude is None:
                raise ValueError("reference_latitude must be provided.")
            else:
                self.reference_latitude = reference_latitude
            if first_standard_parallel is None:
                raise ValueError("first_standard_parallel must be provided.")
            else:
                self.first_standard_parallel = first_standard_parallel
            if second_standard_parallel is None:
                raise ValueError("second_standard_parallel must be provided.")
            else:
                self.second_standard_parallel = second_standard_parallel
            if false_easting is None:
                raise ValueError("false_easting must be provided.")
            else:
                self.false_easting = false_easting
            if false_northing is None:
                raise ValueError("false_northing must be provided.")
            else:
                self.false_northing = false_northing
            self.year = year

        if shapefile_data is None:
            if shapefile_path is None:
                raise ValueError("shapefile_path or shapefile_data must be provided.")
            self.shapefile_data = self.load_census_boundaries(shapefile_path)
        else:
            self.shapefile_data = shapefile_data

    def load_census_boundaries(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """Load the data from the census boundaries shapefile.

        Args:
            shapefile_path: Full path for the shapefile containing the census division boundaries.

        Returns:
            A data frame containing the census division boundaries.
        """
        df = gpd.read_file(shapefile_path)
        df = df.astype({"CDUID": int})
        return df

    def get_census_division_from_lat_lon(self, longitude: float, latitude: float) -> CensusDivision:
        """Given a latitude and longitude, find the corresponding census division.

        Args:
            longitude: the longitude.
            latitude: the latitude.
        """
        point = self.get_lambert_conformal_from_lat_lon(
            λ=longitude,
            φ=latitude,
            λ_0=self.reference_longitude,
            φ_0=self.reference_latitude,
            φ_1=self.first_standard_parallel,
            φ_2=self.second_standard_parallel,
            x_0=self.false_easting,
            y_0=self.false_northing
        )
        point = shapely.geometry.Point(point)

        is_point_in_polygon = False
        name = ""
        cduid = 0

        for row_index in range(self.shapefile_data.shape[0]):
            row = self.shapefile_data.iloc[row_index]
            polygon = row["geometry"]
            if self.point_in_polygon(point, polygon):
                name = row["CDNAME"]
                cduid = row["CDUID"]
                is_point_in_polygon = True
                break

        if not is_point_in_polygon:
            raise ValueError("Could not find point in any of the federal electoral districts.")
        else:
            census_division = CensusDivision(
                name=name,
                cduid=cduid,
                year=self.year
            )
            return census_division

    def point_in_polygon(
        self, point: shapely.Point | tuple, polygon: shapely.Polygon | shapely.MultiPolygon
    ):
        """Determine whether or not a given point is within a polygon.

        Args:
            point: The point we want to check.
            polygon: A polygon object containing the points of its boundary.
        """
        point = shapely.geometry.Point(point)

        if isinstance(polygon, shapely.Polygon):
            return polygon.contains(point)
        else:
            is_in_polygon = False
            for sub_polygon in polygon.geoms:
                if self.point_in_polygon(point, sub_polygon):
                    is_in_polygon = True
                    break
            return is_in_polygon

    def get_lambert_conformal_from_lat_lon(
        self,
        λ: float,
        φ: float,
        λ_0: float,
        φ_0: float,
        φ_1: float,
        φ_2: float,
        x_0: float,
        y_0: float
    ) -> tuple[float, float]:
        """Given a latitude and longitude, find the Lambert Conformal Conic projection coordinates.

        See: 
        
        1. `Lambert Conformal Conic Transformations
           <https://www.linz.govt.nz/guidance/geodetic-system/understanding-coordinate-conversions/projection-conversions/lambert-conformal-conic-geographic-transformation-formulae>`_
        2. `Wikipedia: Geodetic Reference System 1980
           <https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980>`_.

        Args:
            λ: the longitude.
            φ: the latitude.
            λ_0: the reference longitude.
            φ_0: the reference latitude.
            φ_1: the first standard parallel in degrees.
            φ_2: the second standard parallel in degrees.
            x_0: false easting.
            y_0: false northing.

        Returns:
            The Lambert Conformal Conic projection coordinates.
        """
        λ = np.deg2rad(λ)
        φ = np.deg2rad(φ)
        λ_0 = np.deg2rad(λ_0)
        φ_0 = np.deg2rad(φ_0)
        φ_1 = np.deg2rad(φ_1)
        φ_2 = np.deg2rad(φ_2)

        R = 6378137  # Radius of Earth

        f = 0.003352810681183637418  # flattening
        e = np.sqrt(2*f - f**2)  # eccentricity

        m_1 = np.cos(φ_1)/(np.sqrt(1 - e**2*np.sin(φ_1)**2))
        m_2 = np.cos(φ_2)/(np.sqrt(1 - e**2*np.sin(φ_2)**2))
        t = np.tan(np.pi/4 - φ/2) / ((1 - e*np.sin(φ))/(1 + e*np.sin(φ)))**(e/2)
        t_0 = np.tan(np.pi/4 - φ_0/2) / ((1 - e*np.sin(φ_0))/(1 + e*np.sin(φ_0)))**(e/2)
        t_1 = np.tan(np.pi/4 - φ_1/2) / ((1 - e*np.sin(φ_1))/(1 + e*np.sin(φ_1)))**(e/2)
        t_2 = np.tan(np.pi/4 - φ_2/2) / ((1 - e*np.sin(φ_2))/(1 + e*np.sin(φ_2)))**(e/2)

        if φ_1 == φ_2:
            n = np.sin(φ_1)
        else:
            n = (np.log(m_1) - np.log(m_2))/(np.log(t_1) - np.log(t_2))

        F = m_1 / (n*t_1**n)
        ρ_0 = R*F*t_0**n
        ρ = R*F*t**n

        x = x_0 + ρ*np.sin(n*(λ - λ_0))
        y = y_0 + ρ_0 - ρ*np.cos(n*(λ - λ_0))

        return x, y
