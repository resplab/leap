import pytest
import pathlib
import json
import pandas as pd
from leap.census_division import CensusDivision, CensusBoundaries
from leap.utils import PROCESSED_DATA_PATH


@pytest.mark.parametrize(
    "cduid, name, year, province, census_table",
    [
        (None, None, 2021, "CA", None),
    ]
)
def test_census_division_constructor(cduid, name, year, province, census_table):
    census_division = CensusDivision(
        cduid=cduid, name=name, year=year, province=province, census_table=census_table
    )

    if cduid is not None:
        assert census_division.cduid == cduid
    else:
        assert isinstance(census_division.cduid, int)
    if name is not None:
        assert census_division.name == name
    else:
        assert isinstance(census_division.name, str)
    assert census_division.year == year
    if census_table is not None:
        pd.testing.assert_frame_equal(census_division.census_table, census_table)


@pytest.mark.parametrize(
    (
        "shapefile_data, shapefile_path, metadata_path, year, reference_longitude,"
        "reference_latitude, first_standard_parallel, second_standard_parallel, false_easting,"
        "false_northing"
    ),
    [
        (
            None,
            pathlib.Path(
                PROCESSED_DATA_PATH,
                "census_divisions/census_division_boundaries/lcd_000b21a_e.shp"
            ),
            pathlib.Path(
                PROCESSED_DATA_PATH,
                "census_divisions/census_division_boundaries/lcd_000b21a_e_metadata.json"
            ),
            2021,
            None,
            None,
            None,
            None,
            None,
            None
        ),
    ]
)
def test_census_boundaries_constructor(
    shapefile_data, shapefile_path, metadata_path, year, reference_longitude,
    reference_latitude, first_standard_parallel, second_standard_parallel, false_easting,
    false_northing
):
    census_boundaries = CensusBoundaries(
        shapefile_data=shapefile_data,
        shapefile_path=shapefile_path,
        metadata_path=metadata_path,
        year=year,
        reference_longitude=reference_longitude,
        reference_latitude=reference_latitude,
        first_standard_parallel=first_standard_parallel,
        second_standard_parallel=second_standard_parallel,
        false_easting=false_easting,
        false_northing=false_northing
    )
    assert census_boundaries.year == year
    assert census_boundaries.shapefile_data is not None

    if metadata_path is not None:
        with open(metadata_path) as file:
            metadata = json.load(file)
        assert census_boundaries.reference_longitude == metadata["reference_longitude"]
        assert census_boundaries.reference_latitude == metadata["reference_latitude"]
        assert census_boundaries.first_standard_parallel == metadata["first_standard_parallel"]
        assert census_boundaries.second_standard_parallel == metadata["second_standard_parallel"]
        assert census_boundaries.false_easting == metadata["false_easting"]
        assert census_boundaries.false_northing == metadata["false_northing"]
    else:
        assert census_boundaries.metadata_path is None
        assert census_boundaries.reference_longitude == reference_longitude
        assert census_boundaries.reference_latitude == reference_latitude
        assert census_boundaries.first_standard_parallel == first_standard_parallel
        assert census_boundaries.second_standard_parallel == second_standard_parallel
        assert census_boundaries.false_easting == false_easting
        assert census_boundaries.false_northing == false_northing


@pytest.mark.parametrize(
    (
        "shapefile_path, metadata_path, year, latitude, longitude, name, cduid"
    ),
    [
        (
            pathlib.Path(
                PROCESSED_DATA_PATH,
                "census_divisions/census_division_boundaries/lcd_000b21a_e.shp"
            ),
            pathlib.Path(
                PROCESSED_DATA_PATH,
                "census_divisions/census_division_boundaries/lcd_000b21a_e_metadata.json"
            ),
            2021,
            49.262580,
            -123.118720,
            "Greater Vancouver",
            5915
        ),
    ]
)
def test_census_boundaries_get_census_division_from_lat_lon(
    shapefile_path, metadata_path, year, latitude, longitude, name, cduid
):
    census_boundaries = CensusBoundaries(
        shapefile_path=shapefile_path, metadata_path=metadata_path, year=year
    )
    census_division = census_boundaries.get_census_division_from_lat_lon(
        longitude=longitude, latitude=latitude
    )
    assert census_division.name == name
    assert census_division.cduid == cduid
    assert census_division.year == year


@pytest.mark.parametrize(
    "shapefile_path, metadata_path",
    [
        (
            pathlib.Path(
                PROCESSED_DATA_PATH,
                "census_divisions/census_division_boundaries/lcd_000b21a_e.shp"
            ),
            pathlib.Path(
                PROCESSED_DATA_PATH,
                "census_divisions/census_division_boundaries/lcd_000b21a_e_metadata.json"
            )
        )
    ]
)
@pytest.mark.parametrize(
    (
        "year, longitude, latitude, cduid, is_in_polygon"
    ),
    [
        (2021, 4017906.677490763, 2005298.410868233, 5915, True),
        (2021, 4017906.677490763, 1005298.410868233, 5915, False)
    ]
)
def test_census_boundaries_point_in_polygon(
    shapefile_path, metadata_path, year, longitude, latitude, cduid, is_in_polygon
):
    census_boundaries = CensusBoundaries(
        shapefile_path=shapefile_path, metadata_path=metadata_path, year=year
    )
    df = census_boundaries.shapefile_data
    polygon = df[df["CDUID"] == cduid].geometry.values[0]
    assert census_boundaries.point_in_polygon((longitude, latitude), polygon) == is_in_polygon
