import pytest
import datetime as dt
from leap.immigration import Immigration
from leap.utils import round_number


@pytest.mark.parametrize(
    (
        "min_timepoint, timepoint, age, max_age, sex, province, projection_scenario,"
        "prop_immigrants_birth, prop_immigrants_timepoint"
    ),
    [
        (
            dt.datetime(2024, 1, 1),
            dt.datetime(2026, 1, 1),
            4,
            111,
            0,
            "BC",
            "LG",
            0.004955,
            0.004911
        ),
        (
            dt.datetime(2024, 1, 1),
            dt.datetime(2025, 1, 1),
            4,
            111,
            1,
            "BC",
            "LG",
            0.007408,
            0.007319
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
            dt.datetime(2024, 1, 1),
            dt.datetime(2025, 1, 1),
            111,
            "BC",
            "LG",
            1000,
            1013
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
