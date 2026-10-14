import pytest
import pickle
import datetime as dt
from leap.utils import date_range, TimeDelta


@pytest.mark.parametrize(
    "start, stop, time_delta",
    [
        (dt.datetime(2000, 1, 1), dt.datetime(2066, 1, 1), TimeDelta(months=1)),
        (dt.datetime(2000, 1, 1), dt.datetime(2066, 1, 1), TimeDelta(years=1)),
    ]
)
def test_date_range(start, stop, time_delta):
    timepoints = list(date_range(start, stop, time_delta))
    n_intervals = TimeDelta(years=1) // time_delta
    assert len(timepoints) == 66 * n_intervals
    # timepoints stay on the first of the month, with no drift
    assert all(t.day == 1 and t.time() == dt.time(0, 0) for t in timepoints)
    # timepoints are passed between processes in the simulation, so must survive pickling
    assert pickle.loads(pickle.dumps(timepoints)) == timepoints
