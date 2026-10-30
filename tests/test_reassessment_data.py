import pytest
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from leap.data_generation.reassessment_data import get_asthma_df, get_reassessment_data
from leap.logger import get_logger
from leap.utils import TimeDelta, Age, date_range, get_time_delta_tag, get_data_path

logger = get_logger(__name__)
logger.setLevel("INFO")



@pytest.fixture
def df_asthma_dict():
    time_deltas = [TimeDelta(years=1), TimeDelta(months=1)]
    min_timepoint = dt.datetime(2025, 1, 1)
    max_timepoint = dt.datetime(2027, 1, 1)
    df_asthma_dict = {}
    for time_delta in time_deltas:
        n_intervals = TimeDelta(years=1) // time_delta
        df = pd.DataFrame(
            list(itertools.product(
                list(date_range(min_timepoint, max_timepoint + time_delta, time_delta)),
                [Age(value=x) for x in np.arange(0, 4, 1/n_intervals)],
                ["F", "M"]
            )),
            columns=[
                "timepoint", "age", "sex"
            ]
        )
        df["incidence"] = np.random.uniform(0.0001, 0.01, df.shape[0])
        df["prevalence"] = np.random.uniform(0.0001, 0.01, df.shape[0])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        key = time_delta.to_isoformat()
        df_asthma_dict[key] = df.copy()
    return df_asthma_dict


@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1)),
        (TimeDelta(months=1))
    ]
)
@pytest.mark.parametrize(
    "min_timepoint, max_timepoint, min_age, max_age, max_asthma_age, stabilization_timepoint",
    [
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2065, 1, 1),
            0,
            100,
            18,
            dt.datetime(2000, 1, 1)
        )
    ]
)
def test_get_asthma_df(
    time_delta, min_timepoint, max_timepoint, min_age, max_age, max_asthma_age, stabilization_timepoint
):
    df = get_asthma_df(
        time_delta=time_delta,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
        min_age=min_age,
        max_age=max_age,
        max_asthma_age=max_asthma_age,
        stabilization_timepoint=stabilization_timepoint
    )
    assert not df.empty
    assert set(df.columns) == set(["age", "sex", "timepoint", "prevalence", "incidence"])


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