import pytest
import datetime as dt
import numpy as np
import pandas as pd
from leap.logger import get_logger
from leap.utils import TimeDelta, date_range, get_time_delta_tag, get_data_path

logger = get_logger(__name__)
logger.setLevel("INFO")


@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1)),
        (TimeDelta(months=1))
    ]
)
def test_processed_data(time_delta):
    time_delta_tag = get_time_delta_tag(time_delta)
    df = pd.read_csv(
        get_data_path(f"processed_data/{time_delta_tag}/asthma_reassessment.csv"),
        parse_dates=["timepoint"]
    )

    # Check that the timepoints in the dataframe form a complete sequence with the given time delta
    timepoints = df["timepoint"].unique()
    min_timepoint = timepoints.min()
    max_timepoint = timepoints.max()
    pd.testing.assert_series_equal(
        pd.Series(timepoints),
        pd.Series(list(date_range(start=min_timepoint, stop=max_timepoint + time_delta, step=time_delta))),
        check_index=False,
        check_names=False
    )

    # Check that the dataframe contains the expected columns
    assert set(df.columns) == set(["age", "sex", "timepoint", "province", "prob"])

    # Check that the sex is one of "M" or "F"
    assert df["sex"].isin(["M", "F"]).all()

    # Check that the age column contains non-negative integers
    assert (df["age"] >= 0).all()
    assert pd.api.types.is_integer_dtype(df["age"])

    # Check that probability is a number in [0, 1]
    assert (df["prob"] >= 0).all()
    assert (df["prob"] <= 1).all()