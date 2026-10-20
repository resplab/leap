import pytest
import datetime as dt
import pandas as pd
from leap.immigration import Immigration
from leap.utils import round_number, Timepoint


@pytest.mark.parametrize(
    (
        "min_timepoint, timepoint, age, max_age, sex, province, projection_scenario,"
        "prop_immigrants_birth, prop_immigrants_timepoint"
    ),
    [
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2005, 1, 1),
            4,
            111,
            "M",
            "BC",
            "LG",
            0.006876,
            0.008984
        ),
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2003, 1, 1),
            4,
            111,
            "F",
            "BC",
            "LG",
            0.002131,
            0.005572
        ),
    ]
)
def test_immigration_constructor(
    min_timepoint, timepoint, age, max_age, sex, province, projection_scenario,
    prop_immigrants_birth, prop_immigrants_timepoint
):
    immigration = Immigration(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario,
        max_age=max_age
    )
    df = immigration.get_table_group((timepoint))
    row = df[(df["age"] == age) & (df["sex"] == sex)]
    assert round_number(row["prop_immigrants_birth"].values[0], sigdigits=4) == prop_immigrants_birth
    assert round_number(row["prop_immigrants_timepoint"].values[0], sigdigits=4) == prop_immigrants_timepoint


@pytest.mark.parametrize(
    (
        "min_timepoint, timepoint, max_age, province, projection_scenario,"
        "num_new_born, num_new_immigrants"
    ),
    [
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2003, 1, 1),
            111,
            "BC",
            "LG",
            1000,
            383
        )
    ]
)
def test_immigration_get_num_new_immigrants(
    min_timepoint, timepoint, max_age, province, projection_scenario, num_new_born,
    num_new_immigrants
):
    immigration = Immigration(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario,
        max_age=max_age
    )
    assert immigration.get_num_new_immigrants(num_new_born, timepoint) == num_new_immigrants


@pytest.mark.parametrize(
    "timepoint",
    [
        dt.datetime(2030, 1, 1),
        pd.Timestamp(2030, 1, 1),
        Timepoint(2030, 1, 1),
        dt.datetime(2030, 1, 1, 0, 0, 0, 15),
        dt.datetime(2029, 12, 31, 23, 59, 59, 999985),
        dt.datetime(2030, 1, 15),
    ]
)
def test_immigration_get_table_group(timepoint):
    # lookups match the month's group whatever the key type, and despite small differences in time
    immigration = Immigration(
        min_timepoint=dt.datetime(2000, 1, 1), province="BC", projection_scenario="LG"
    )
    expected = immigration.get_table_group(Timepoint(2030, 1, 1))
    assert immigration.get_table_group(timepoint).equals(expected)
