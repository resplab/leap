import pytest
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from leap.data_generation.migration_data import get_delta_n,  \
    load_population_data, load_migration_data, MIN_TIMEPOINT
from leap.logger import get_logger
from leap.utils import TimeDelta, date_range, PROJECTION_SCENARIOS, PROVINCE_MAP, Age

logger = get_logger(__name__)


@pytest.fixture
def df_populations():
    time_deltas = [TimeDelta(years=1), TimeDelta(months=1)]
    min_timepoint = dt.datetime(2025, 1, 1)
    max_timepoint = dt.datetime(2027, 1, 1)
    population_dict = {}
    for time_delta in time_deltas:
        n_intervals = TimeDelta(years=1) // time_delta
        df = pd.DataFrame(
            list(itertools.product(
                ["BC", "CA"],
                ["past", "LG"],
                list(date_range(min_timepoint, max_timepoint + time_delta, time_delta)),
                [Age(value=x) for x in np.arange(0, 4, 1/n_intervals)],
                ["F", "M"]
            )),
            columns=[
                "province", "projection_scenario", "timepoint", "age", "sex"
            ]
        )
        df["n"] = np.random.randint(1000, 10000, df.shape[0])

        df.loc[
            (df["timepoint"] == max_timepoint) &
            (df["sex"] == "F") &
            (df["age"] == 2.0) &
            (df["province"] == "BC") &
            (df["projection_scenario"] == "LG"), "n"
        ] = 1000
        df.loc[
            (df["timepoint"] == max_timepoint - time_delta) &
            (df["sex"] == "F") &
            (df["age"] == 2.0 - time_delta.total_years()) &
            (df["province"] == "BC") &
            (df["projection_scenario"] == "LG"), "n"
        ] = 1500
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        key = time_delta.to_isoformat()
        population_dict[key] = df.copy()
    return population_dict


@pytest.fixture
def life_tables():
    time_deltas = [TimeDelta(years=1), TimeDelta(months=1)]
    min_timepoint = dt.datetime(2025, 1, 1)
    max_timepoint = dt.datetime(2027, 1, 1)
    life_table_dict = {}
    for time_delta in time_deltas:
        n_intervals = TimeDelta(years=1) // time_delta
        life_table = pd.DataFrame(
            list(itertools.product(
                ["BC", "CA"],
                ["past", "LG"],
                [Age(value=x) for x in np.arange(0, 4, 1/n_intervals)],
                ["F", "M"],
                list(date_range(min_timepoint, max_timepoint + time_delta, time_delta))
            )),
            columns=["province", "projection_scenario", "age", "sex", "timepoint"]
        )
        life_table["prob_death"] = np.random.sample(life_table.shape[0]) / 1000.0
        life_table.loc[
            (life_table["timepoint"] == max_timepoint - time_delta) &
            (life_table["sex"] == "F") &
            (life_table["age"] == round(2.0 - time_delta.total_years(), 6)) &
            (life_table["province"] == "BC") &
            (life_table["projection_scenario"] == "LG"), "prob_death"
        ] = 0.5
        key = time_delta.to_isoformat()
        life_table_dict[key] = life_table
    return life_table_dict


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
def test_load_migration_data(df_populations, life_tables, time_delta):
    df_population = df_populations[time_delta.to_isoformat()]
    life_table = life_tables[time_delta.to_isoformat()]
    df = load_migration_data(df_population, life_table, time_delta)
    assert set(df.columns) == set([
        "province", "projection_scenario", "timepoint", "age", "sex",
        "n_emigrants", "n", "prob_emigration", "n_birth",
        "delta_n", "prop_migrants_birth", "prop_immigrants_timepoint", "prop_emigrants_timepoint",
        "prob_death", "n_immigrants", "n_immigrants_timepoint", "n_emigrants_timepoint"
    ])
    assert not df.empty
    assert df["province"].isin(list(PROVINCE_MAP.values())).all()
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS).all()
    assert df["sex"].isin(["F", "M"]).all()
    row = df.loc[
        (df["timepoint"] == dt.datetime(2027, 1, 1)) &
        (df["sex"] == "F") &
        (df["age"] == 2.0) &
        (df["province"] == "BC") &
        (df["projection_scenario"] == "LG")
    ]
    assert row["n"].iloc[0] == 1000
    assert row["delta_n"].iloc[0] == 1000 - 1500 * (1 - 0.5)

