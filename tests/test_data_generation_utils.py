    

import pytest
import datetime as dt
from leap.utils import date_range, TimeDelta
from leap.data_generation.utils import convert_timepoint_to_numeric, convert_numeric_to_timepoint


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


def test_convert_timepoint_round_trip():
    # without rounding, 58 of these monthly timepoints came back a few microseconds off, and
    # 19 landed in the previous day, month, or year
    for timepoint in date_range(dt.datetime(2000, 1, 1), dt.datetime(2066, 1, 1), TimeDelta(months=1)):
        result = convert_numeric_to_timepoint(convert_timepoint_to_numeric(timepoint))
        assert result == timepoint
        assert type(result) is dt.datetime
