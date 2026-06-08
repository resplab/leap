import pytest
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from leap.data_generation.death_data import load_past_death_data, \
    load_projected_death_data, get_prob_death_projected, get_projected_life_table_single_timepoint, \
    MIN_TIMEPOINT, TIME_DELTA_OD
from leap.logger import get_logger
from leap.utils import TimeDelta, PROJECTION_SCENARIOS, MORTALITY_SCENARIOS, PROVINCE_MAP

logger = get_logger(__name__)


@pytest.fixture
def life_table():
    life_table = pd.DataFrame(
        list(itertools.product(
            list(PROVINCE_MAP.values())[0:2],
            np.arange(0, 4, 0.25),
            ["F", "M"],
            [MIN_TIMEPOINT]
        )),
        columns=["province", "age", "sex", "timepoint"]
    )
    life_table["prob_death"] = np.random.sample(life_table.shape[0]) / 1000.0
    life_table["se"] = np.random.sample(life_table.shape[0]) / 10000.0
    return life_table


@pytest.mark.parametrize(
    "prob_death_initial, timepoint_initial, timepoint, beta_time",
    [
        (
            1.0,
            dt.datetime(2020, 1, 1),
            dt.datetime(2021, 1, 1),
            0.1
        )
    ]
)
def test_get_prob_death_projected(
    prob_death_initial, timepoint_initial, timepoint, beta_time
):
    prob_death = get_prob_death_projected(
        prob_death=prob_death_initial,
        timepoint_initial=timepoint_initial,
        timepoint=timepoint,
        beta_time=beta_time
    )
    assert prob_death >= 0.0
    assert prob_death <= 1.0


@pytest.mark.parametrize(
    "sex, province, timepoint",
    [
        ("F", "BC", dt.datetime(2026, 1, 1))
    ]
)
def test_get_projected_life_table_single_timepoint(life_table, sex, province, timepoint):
    projected_life_table = get_projected_life_table_single_timepoint(
        beta_time=0.1,
        life_table=life_table,
        timepoint_initial=life_table["timepoint"].iloc[0],
        timepoint=timepoint,
        sex=sex,
        province=province
    )
    assert projected_life_table["timepoint"].nunique() == 1
    assert projected_life_table["timepoint"].iloc[0] == timepoint
    assert projected_life_table["sex"].nunique() == 1
    assert projected_life_table["sex"].iloc[0] == sex
    assert projected_life_table["province"].nunique() == 1
    assert projected_life_table["province"].iloc[0] == province
    assert set(projected_life_table.columns) == set(
        ["province", "age", "sex", "timepoint", "prob_death", "se"]
    )
    assert set(projected_life_table["age"].unique()) == set(life_table["age"].unique())
    assert projected_life_table["prob_death"].between(0.0, 1.0).all()
    assert projected_life_table["se"].between(0.0, 1.0).all()


@pytest.mark.parametrize(
    "time_delta, expected_rows",
    [
        (
            TimeDelta(years=1),
            [(
                dt.datetime(2019, 1, 1), "BC", 2, "F",
                0.00015
            )],
        ),
        (
            TimeDelta(months=1),
            [(
                dt.datetime(2019, 1, 1), "BC", 2, "F",
                0.00015 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds()
            )],
        ),
    ]
)
def test_load_past_death_data(time_delta, expected_rows):
    df = load_past_death_data(time_delta)
    assert df["timepoint"].min() >= MIN_TIMEPOINT
    assert df["sex"].isin(["M", "F"]).all()
    assert set(df.columns) == set(["province", "age", "sex", "prob_death", "timepoint", "se"])
    assert df["province"].isin(PROVINCE_MAP.values()).all()
    for row in expected_rows:
        assert df.loc[
            (df["timepoint"] == row[0]) &
            (df["province"] == row[1]) &
            (df["age"] == row[2]) &
            (df["sex"] == row[3])
        ].iloc[0]["prob_death"] == row[4]


def test_load_projected_death_data():
    df = load_projected_death_data()
    assert df["life_expectancy"].min() > 50.0
    assert df["life_expectancy"].max() < 120.0
    assert df["sex"].isin(["M", "F"]).all()
    assert set(df.columns) == set([
        "province", "sex", "life_expectancy", "timepoint",
        "projection_scenario", "mortality_scenario"
    ])
    assert df["province"].isin(PROVINCE_MAP.values()).all()
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS).all()
    assert df["mortality_scenario"].isin(MORTALITY_SCENARIOS).all()

