import pytest
import datetime as dt
import pandas as pd
from leap.immigration import Immigration
from leap.utils import round_number


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
            1,
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
            0,
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
    df = immigration.table.get_group((timepoint))
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


def test_immigration_timepoint_key_types():
    # the table is grouped by pandas Timestamps; lookups must work with plain datetimes too
    immigration = Immigration(
        min_timepoint=dt.datetime(2000, 1, 1), province="BC", projection_scenario="LG"
    )
    timepoint = dt.datetime(2030, 1, 1)
    assert immigration.table.get_group(timepoint).equals(
        immigration.table.get_group(pd.Timestamp(timepoint))
    )
    assert immigration.get_num_new_immigrants(1000, timepoint) == \
        immigration.get_num_new_immigrants(1000, pd.Timestamp(timepoint))
