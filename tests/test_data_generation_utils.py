    

import pytest
import datetime as dt
from leap.data_generation.utils import convert_timepoint_to_numeric, convert_numeric_to_timepoint


@pytest.mark.parametrize(
    "timepoint, expected_value",
    [
        (dt.datetime(2024, 1, 1), 2023.958932238193),
        (dt.datetime(2066, 12, 1), 2066.874743326489),
        (dt.datetime(1, 1, 1), 1.002053388090349)
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
        (1.002053388090349, dt.datetime(1, 1, 1))
    ]
)
def test_convert_numeric_to_timepoint(timepoint, expected_value):
    result = convert_numeric_to_timepoint(timepoint)
    assert result == expected_value