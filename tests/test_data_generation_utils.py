    

import pytest
import datetime as dt
from leap.data_generation.utils import convert_timepoint_to_numeric, convert_numeric_to_timepoint, \
    get_max_timepoint
from leap.utils import TimeDelta


@pytest.mark.parametrize(
    "timepoint, expected_value",
    [
        (dt.datetime(2024, 1, 1), 2023.958932238193),
        (dt.datetime(2066, 12, 1), 2066.874743326489),
        (dt.datetime(1, 1, 1), 1.002053388090349),
        (dt.datetime(2000, 1, 1), 1999.958932238193),
        (dt.datetime(2000, 2, 1), 2000.043805612594),
    ]
)
def test_convert_timepoint_to_numeric(timepoint, expected_value):
    result = convert_timepoint_to_numeric(timepoint)
    assert result == expected_value


@pytest.mark.parametrize(
    "timepoint, expected_value",
    [
        (2023.958932238193, dt.datetime(2024, 1, 1)),
        (2066.874743326489, dt.datetime(2066, 12, 1)),
        (1.002053388090349, dt.datetime(1, 1, 1)),
        (1999.958932238193, dt.datetime(2000, 1, 1)),
        (2000.043805612594, dt.datetime(2000, 2, 1)),
    ]
)
def test_convert_numeric_to_timepoint(timepoint, expected_value):
    result = convert_numeric_to_timepoint(timepoint)
    assert result == expected_value



@pytest.mark.parametrize(
    "min_timepoint, max_timepoint, time_delta, expected_max_timepoint",
    [
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2024, 12, 31),
            TimeDelta(years=1),
            dt.datetime(2024, 1, 1)
        ),
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2024, 12, 31),
            TimeDelta(months=1),
            dt.datetime(2024, 12, 1)
        ),
    ]
)
def test_get_max_timepoint(min_timepoint, max_timepoint, time_delta, expected_max_timepoint):

    max_timepoint = get_max_timepoint(min_timepoint, max_timepoint, time_delta)
    assert max_timepoint == expected_max_timepoint