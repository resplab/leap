import pytest
import datetime as dt
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import itertools
from leap.data_generation.occurrence_data import load_asthma_df, generate_occurrence_model, \
    MIN_TIMEPOINT
from leap.logger import get_logger
from leap.utils import TimeDelta, date_range, PROJECTION_SCENARIOS, Age

logger = get_logger(__name__)


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