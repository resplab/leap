import pytest
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from leap.data_generation.migration_data import get_delta_n,  \
    load_population_data, load_migration_data, MIN_TIMEPOINT
from leap.logger import get_logger
from leap.utils import TimeDelta, date_range, PROJECTION_SCENARIOS, PROVINCE_MAP

logger = get_logger(__name__)



@pytest.fixture
def df_projection():
    df_projection = pd.DataFrame(
        list(itertools.product(
            ["BC", "CA"],
            PROJECTION_SCENARIOS,
            list(date_range(dt.datetime(2025, 1, 1), dt.datetime(2027, 1, 1), TimeDelta(months=1))),
            np.arange(0, 4, 1/12),
            ["F", "M"]
        )),
        columns=[
            "province", "projection_scenario", "timepoint", "age", "sex"
        ]
    )
    df_projection["n"] = np.random.randint(1000, 10000, df_projection.shape[0])
    df_projection["prob_death"] = np.random.uniform(0.0001, 0.01, df_projection.shape[0])
    return df_projection


@pytest.mark.parametrize(
    "n, n_prev, prob_death",
    [
        (1000, 1100, 0.01),
        (1000, 900, 0.01)
    ]
)
def test_get_delta_n(n, n_prev, prob_death):
    delta_n = get_delta_n(n=n, n_prev=n_prev, prob_death=prob_death)
    assert np.abs(delta_n) <= n


@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1)),
        (TimeDelta(months=1))
    ]
)
def test_load_population_data(time_delta):
    df = load_population_data(time_delta)
    assert set(df.columns) == set(
        ["province", "projection_scenario", "timepoint", "age", "sex", "n"]
    )
    assert df["province"].isin(list(PROVINCE_MAP.values())).all()
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS).all()
    assert df["sex"].isin(["F", "M"]).all()


@pytest.mark.parametrize(
    "time_delta",
    [
        (TimeDelta(years=1)),
        (TimeDelta(months=1))
    ]
)
def test_load_migration_data(time_delta):
    df = load_migration_data(time_delta)
    assert set(df.columns) == set(
        ["province", "projection_scenario", "timepoint", "age", "sex", "n_prev", "prob_death_prev"]
    )
    assert df["province"].isin(list(PROVINCE_MAP.values())).all()
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS).all()
    assert df["sex"].isin(["F", "M"]).all()

