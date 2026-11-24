import pytest
import datetime as dt
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import itertools
from leap.data_generation.occurrence_data import load_asthma_df, generate_occurrence_model, \
    generate_prevalence_model, generate_incidence_model, get_predicted_data, MIN_TIMEPOINT
from leap.logger import get_logger
from leap.occurrence import MIN_ASTHMA_AGE
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
    "occ_type, formula, min_timepoint, max_timepoint",
    [
        ("incidence", "incidence ~ sex*timepoint", MIN_TIMEPOINT, dt.datetime(2027, 1, 1)),
        ("prevalence", "prevalence ~ sex*poly(timepoint, degree=2)", MIN_TIMEPOINT, dt.datetime(2027, 1, 1))
    ]
)
def test_generate_occurrence_model(
    df_asthma, time_delta, occ_type, formula, min_timepoint, max_timepoint
):
    results = generate_occurrence_model(
        df_asthma=df_asthma,
        formula=formula,
        occ_type=occ_type,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
        time_delta=time_delta,
        maxiter=10
    )
    assert isinstance(results, GLMResultsWrapper)


@pytest.mark.parametrize(
    "time_delta, min_timepoint, max_timepoint",
    [
        (TimeDelta(years=1), MIN_TIMEPOINT, dt.datetime(2027, 1, 1))
    ]
)
def test_generate_incidence_model(df_asthma, time_delta, min_timepoint, max_timepoint):
    results = generate_incidence_model(
        df_asthma=df_asthma,
        time_delta=time_delta,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
        maxiter=10
    )
    assert isinstance(results, GLMResultsWrapper)


@pytest.mark.parametrize(
    "time_delta, min_timepoint, max_timepoint",
    [
        (TimeDelta(years=1), MIN_TIMEPOINT, dt.datetime(2027, 1, 1))
    ]
)
def test_generate_prevalence_model(df_asthma, time_delta, min_timepoint, max_timepoint):
    results, alpha_age, norm2_age, alpha_time, norm2_time = generate_prevalence_model(
        df_asthma=df_asthma,
        time_delta=time_delta,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
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
        df_asthma, time_delta, min_timepoint=min_timepoint, max_timepoint=max_timepoint, maxiter=10
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


# ---------------------------------------------------------------------------------------------
# Annual vs monthly consistency of asthma_occurrence_predictions.csv
#
# Convention: the annual row for year Y corresponds to January of year Y on the monthly scale.
#
# * incidence: the monthly CSV stores the monthly probability
#   p_k = 1 - (1 - pbar_k) ** (1 / 12), where pbar_k is the GLM-predicted (annual-scale)
#   incidence at month k. Cumulative incidence since January, cum = 1 - cumprod(1 - p_k), is
#   compared with the annual incidence.
# * prevalence: a point-in-time state, so the monthly CSV stores the GLM-predicted (annual-scale)
#   prevalence at each month, with no conversion.
#
# Ages above 63 (the midpoint of the oldest training age group) are excluded: predictions are capped at age 63 at runtime. Incidence at age 3 is excluded
# because it contains the age-3 prevalence.
# ---------------------------------------------------------------------------------------------

MAX_PREDICTION_AGE = 63
MIN_INCIDENCE_AGE = MIN_ASTHMA_AGE + 1

# Floating-point tolerance for checks that hold exactly in theory. Converting p -> 1 - (1 - p) ** (1/12)
# and back computes 1 - p for small p, which loses precision (relative error ~1e-12 at p ~ 6e-4).
EXACT_RTOL = 1e-9

# The prevalence GLM has a quadratic time term, so within a year the curve can reach a peak or
# low point and move slightly outside the range of the annual values for years Y and Y + 1. This
# is a chosen sanity tolerance, not a derived one.
PREVALENCE_BOUND_RTOL = 0.02


def load_occurrence_predictions(time_delta: TimeDelta) -> pd.DataFrame:
    """Load ``asthma_occurrence_predictions.csv`` for a given time delta, with a ``year`` column."""
    df = pd.read_csv(
        get_data_path(
            f"processed_data/{get_time_delta_tag(time_delta)}/asthma_occurrence_predictions.csv"
        ),
        parse_dates=["timepoint"]
    )
    df["year"] = df["timepoint"].dt.year
    df["month"] = df["timepoint"].dt.month
    return df


@pytest.fixture(scope="module")
def df_occ_annual():
    return load_occurrence_predictions(TimeDelta(years=1))


@pytest.fixture(scope="module")
def df_occ_monthly():
    return load_occurrence_predictions(TimeDelta(months=1))


def filter_ages(df: pd.DataFrame, min_age: int) -> pd.DataFrame:
    return df.loc[(df["age"] >= min_age) & (df["age"] <= MAX_PREDICTION_AGE)]


def add_annual_bounds(df: pd.DataFrame, df_annual: pd.DataFrame, col: str) -> pd.DataFrame:
    """Merge the annual values for years Y and Y + 1 onto ``df`` (which has a ``year`` column).

    Adds ``{col}_lower`` and ``{col}_upper``, the smaller and larger of the two annual values.
    Rows without an annual value for Y + 1 are dropped.
    """
    annual = df_annual[["year", "sex", "age", col]]
    annual_next = annual.assign(year=annual["year"] - 1)
    df = df.merge(
        annual.rename(columns={col: f"{col}_annual"}), on=["year", "sex", "age"]
    ).merge(
        annual_next.rename(columns={col: f"{col}_annual_next"}), on=["year", "sex", "age"]
    )
    df[f"{col}_lower"] = np.minimum(df[f"{col}_annual"], df[f"{col}_annual_next"])
    df[f"{col}_upper"] = np.maximum(df[f"{col}_annual"], df[f"{col}_annual_next"])
    return df


def assert_no_rows(df_bad: pd.DataFrame, message: str):
    if not df_bad.empty:
        logger.info(df_bad.head(20))
        raise AssertionError(f"{message} ({df_bad.shape[0]} rows). First rows:\n{df_bad.head(10)}")


def test_monthly_incidence_january_matches_annual(df_occ_annual, df_occ_monthly):
    """Twelve months at January's monthly incidence reproduce the annual incidence for year Y.

    1 - (1 - p_Jan) ** 12 == annual incidence for year Y, for every sex and age.
    """
    df_jan = filter_ages(df_occ_monthly, MIN_INCIDENCE_AGE)
    df_jan = df_jan.loc[df_jan["month"] == 1]
    df = df_jan.merge(
        df_occ_annual[["year", "sex", "age", "incidence"]],
        on=["year", "sex", "age"],
        suffixes=("_monthly", "_annual")
    )
    assert df.shape[0] == df_jan.shape[0], "Every January row should have a matching annual row."

    np.testing.assert_allclose(
        1 - (1 - df["incidence_monthly"]) ** 12,
        df["incidence_annual"],
        rtol=EXACT_RTOL
    )


def test_monthly_incidence_cumulative_within_annual_bounds(df_occ_annual, df_occ_monthly):
    """Cumulative incidence over the 12 months of year Y lies between the annual incidence for
    years Y and Y + 1.

    cum = 1 - cumprod(1 - p_k); its December value, 1 - prod(1 - p_k), is the year's total.
    """
    df = filter_ages(df_occ_monthly, MIN_INCIDENCE_AGE).sort_values(["sex", "age", "timepoint"])
    df["prob_no_diagnosis"] = 1 - df["incidence"]
    df_year = df.groupby(["year", "sex", "age"]).agg(
        n_months=("month", "nunique"),
        prob_no_diagnosis=("prob_no_diagnosis", "prod")
    ).reset_index()
    df_year = df_year.loc[df_year["n_months"] == 12]
    df_year["cum_december"] = 1 - df_year["prob_no_diagnosis"]

    df_year = add_annual_bounds(df_year, df_occ_annual, "incidence")
    assert not df_year.empty

    df_bad = df_year.loc[
        (df_year["cum_december"] < df_year["incidence_lower"] * (1 - EXACT_RTOL)) |
        (df_year["cum_december"] > df_year["incidence_upper"] * (1 + EXACT_RTOL))
    ]
    assert_no_rows(
        df_bad[["year", "sex", "age", "cum_december", "incidence_lower", "incidence_upper"]],
        "December cumulative incidence is outside the annual incidence for years Y and Y + 1"
    )


def test_monthly_prevalence_january_matches_annual(df_occ_annual, df_occ_monthly):
    """Monthly prevalence in January of year Y equals the annual prevalence for year Y."""
    df_jan = filter_ages(df_occ_monthly, MIN_ASTHMA_AGE)
    df_jan = df_jan.loc[df_jan["month"] == 1]
    df = df_jan.merge(
        df_occ_annual[["year", "sex", "age", "prevalence"]],
        on=["year", "sex", "age"],
        suffixes=("_monthly", "_annual")
    )
    assert df.shape[0] == df_jan.shape[0], "Every January row should have a matching annual row."

    np.testing.assert_allclose(
        df["prevalence_monthly"],
        df["prevalence_annual"],
        rtol=EXACT_RTOL
    )


def test_monthly_prevalence_within_annual_bounds(df_occ_annual, df_occ_monthly):
    """Each monthly prevalence in year Y lies between the annual prevalence for years Y and Y + 1,
    within ``PREVALENCE_BOUND_RTOL``.
    """
    df = add_annual_bounds(
        filter_ages(df_occ_monthly, MIN_ASTHMA_AGE), df_occ_annual, "prevalence"
    )
    assert not df.empty

    df_bad = df.loc[
        (df["prevalence"] < df["prevalence_lower"] * (1 - PREVALENCE_BOUND_RTOL)) |
        (df["prevalence"] > df["prevalence_upper"] * (1 + PREVALENCE_BOUND_RTOL))
    ]
    assert_no_rows(
        df_bad[["timepoint", "sex", "age", "prevalence", "prevalence_lower", "prevalence_upper"]],
        "Monthly prevalence is outside the annual prevalence for years Y and Y + 1"
    )
    