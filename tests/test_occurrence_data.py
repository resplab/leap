import pytest
import datetime as dt
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import itertools
from leap.data_generation.occurrence_data import load_asthma_df, generate_occurrence_model, \
    generate_prevalence_model, generate_incidence_model, get_predicted_data, MIN_TIMEPOINT
from leap.logger import get_logger
from leap.utils import TimeDelta, date_range, get_time_delta_tag, get_data_path

logger = get_logger(__name__)
logger.setLevel("INFO")


@pytest.fixture
def df_asthma():
    df_asthma = pd.DataFrame(
        list(itertools.product(
            list(date_range(dt.datetime(2025, 1, 1), dt.datetime(2027, 1, 1), TimeDelta(months=1))),
            np.arange(0, 4, 1/12),
            ["F", "M"]
        )),
        columns=[
            "timepoint", "age", "sex"
        ]
    )
    df_asthma["incidence"] = np.random.uniform(0.0001, 0.01, df_asthma.shape[0])
    df_asthma["prevalence"] = np.random.uniform(0.0001, 0.01, df_asthma.shape[0])
    return df_asthma


@pytest.mark.parametrize(
    "min_timepoint",
    [
        MIN_TIMEPOINT
    ]
)
def test_load_asthma_df(min_timepoint):
    df = load_asthma_df(min_timepoint=min_timepoint)
    assert set(df.columns) == set(["timepoint", "sex", "age_group", "age", "incidence", "prevalence"])
    assert df["age"].max() <= 65
    assert df["age"].min() >= 3

    # Check that incidence = prevalence at age 3
    pd.testing.assert_series_equal(
        df.loc[df["age"] == 3]["incidence"],
        df.loc[df["age"] == 3]["prevalence"],
        check_names=False
    )

    # Check that prevalence >= incidence
    assert (df["prevalence"] >= df["incidence"]).all()


@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1))
    ]
)
@pytest.mark.parametrize(
    "occ_type, formula",
    [
        ("incidence", "incidence ~ sex*timepoint"),
        ("prevalence", "prevalence ~ sex*poly(timepoint, degree=2)")
    ]
)
def test_generate_occurrence_model(df_asthma, time_delta, occ_type, formula):
    results = generate_occurrence_model(
        df_asthma=df_asthma,
        formula=formula,
        occ_type=occ_type,
        time_delta=time_delta,
        maxiter=10
    )
    assert isinstance(results, GLMResultsWrapper)


@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1))
    ]
)
def test_generate_incidence_model(df_asthma, time_delta):
    results = generate_incidence_model(
        df_asthma=df_asthma,
        time_delta=time_delta,
        maxiter=10
    )
    assert isinstance(results, GLMResultsWrapper)


@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1))
    ]
)
def test_generate_prevalence_model(df_asthma, time_delta):
    results, alpha_age, norm2_age, alpha_time, norm2_time = generate_prevalence_model(
        df_asthma=df_asthma,
        time_delta=time_delta,
        maxiter=10
    )
    assert isinstance(results, GLMResultsWrapper)




@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1)),
        (TimeDelta(months=1))
    ]
)
@pytest.mark.parametrize(
    "min_timepoint, max_timepoint",
    [
        (dt.datetime(1995,12,2), dt.datetime(2000,12,2))
    ]
)
def test_get_predicted_data(df_asthma, time_delta, min_timepoint, max_timepoint):
    model = generate_incidence_model(
        df_asthma, time_delta, maxiter=10
    )
    df = get_predicted_data(
        model=model,
        pred_col="incidence",
        time_delta=time_delta,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint
    )
    assert isinstance(df, pd.DataFrame)
    assert df["timepoint"].min().to_pydatetime() == min_timepoint
    assert df["timepoint"].max().to_pydatetime() == max_timepoint
    grouped_df = df.groupby(["age", "sex"])
    for _, group in grouped_df:
        pd.testing.assert_series_equal(
            group["timepoint"],
            pd.Series(
                list(date_range(start=min_timepoint, stop=max_timepoint + time_delta, step=time_delta))
            ),
            check_index=False,
            check_names=False
        )


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
        get_data_path(f"processed_data/{time_delta_tag}/asthma_occurrence_predictions.csv"),
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
    assert set(df.columns) == set(["age", "sex", "timepoint", "incidence", "prevalence"])

    # Check that the sex is one of "M" or "F"
    assert df["sex"].isin(["M", "F"]).all()

    # Check that the age column contains non-negative integers
    assert (df["age"] >= 0).all()
    assert pd.api.types.is_integer_dtype(df["age"])

    # Check that prevalence >= incidence
    mask = df.loc[df["prevalence"] < df["incidence"]]
    if not mask.empty:
        logger.info(mask)
        raise AssertionError("Prevalence should be greater than or equal to incidence for all rows.")
    