import pytest
import datetime as dt
import pandas as pd
from leap.data_generation.death_data import load_past_death_data, \
    load_projected_death_data, get_prob_death_projected, STARTING_TIMEPOINT
from leap.logger import get_logger
from leap.utils import TimeDelta, PROJECTION_SCENARIOS, MORTALITY_SCENARIOS, PROVINCE_MAP

logger = get_logger(__name__)

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
    "time_delta",
    [
        TimeDelta(years=1),
        TimeDelta(months=1)
    ]
)
def test_load_past_death_data(time_delta):
    df = load_past_death_data(time_delta)
    assert df["timepoint"].min() >= STARTING_TIMEPOINT
    assert df["sex"].isin(["M", "F"]).all()
    assert set(df.columns) == set(["province", "age", "sex", "prob_death", "timepoint", "se"])
    assert df["province"].isin(PROVINCE_MAP.values()).all()


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

