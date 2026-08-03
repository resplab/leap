import pytest
import pathlib
import json
import numpy as np
import datetime as dt
from leap.utils import Timepoint



@pytest.mark.parametrize(
    "year, month, day, hour, minute, second, microsecond",
    [
        (2024, 6, 1, 12, 30, 0, 0),
    ]
)
def test_timepoint_constructor(year, month, day, hour, minute, second, microsecond):
    timepoint = Timepoint(year, month, day, hour, minute, second, microsecond)
    assert timepoint.year == year
    assert timepoint.month == month
    assert timepoint.day == day
    assert timepoint.hour == hour
    assert timepoint.minute == minute
    assert timepoint.second == second
    assert timepoint.microsecond == microsecond


@pytest.mark.parametrize(
    "year, month, day, hour, minute, second, microsecond",
    [
        (2024, 6, 1, 12, 30, 0, 0),
    ]
)
def test_timepoint_from_datetime(year, month, day, hour, minute, second, microsecond):
    dt_obj = dt.datetime(year, month, day, hour, minute, second, microsecond)
    timepoint = Timepoint.from_datetime(dt_obj)
    assert isinstance(timepoint, Timepoint)
    assert timepoint.year == dt_obj.year
    assert timepoint.month == dt_obj.month
    assert timepoint.day == dt_obj.day
    assert timepoint.hour == dt_obj.hour
    assert timepoint.minute == dt_obj.minute
    assert timepoint.second == dt_obj.second
    assert timepoint.microsecond == dt_obj.microsecond