import pytest
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from leap.data_generation.death_data import load_past_death_data, \
    load_projected_death_data, get_prob_death_projected, get_projected_life_table_single_timepoint, \
    get_projected_death_data, MIN_TIMEPOINT, TIME_DELTA_OD
from leap.logger import get_logger
from leap.utils import TimeDelta, PROJECTION_SCENARIOS, MORTALITY_SCENARIOS, PROVINCE_MAP

logger = get_logger(__name__)


@pytest.fixture
def life_table():
    life_table = pd.DataFrame(
        list(itertools.product(
            list(PROVINCE_MAP.values())[0:2],
            PROJECTION_SCENARIOS[0:2],
            np.arange(0, 4, 0.25),
            ["F", "M"],
            [MIN_TIMEPOINT, MIN_TIMEPOINT + TIME_DELTA_OD]
        )),
        columns=["province", "projection_scenario", "age", "sex", "timepoint"]
    )
    mask = (
        ((life_table["projection_scenario"] == "past") & (life_table["timepoint"] > MIN_TIMEPOINT))
        |
        ((life_table["projection_scenario"] != "past") & (life_table["timepoint"] == MIN_TIMEPOINT))
    )
    life_table = life_table.loc[~mask].copy()
    life_table["prob_death"] = np.random.sample(life_table.shape[0]) / 1000.0
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
    life_table_filtered = life_table.loc[
        (life_table["sex"] == sex) &
        (life_table["province"] == province) &
        (life_table["projection_scenario"] == "past")
    ].copy()
    projected_life_table = get_projected_life_table_single_timepoint(
        beta_time=0.1,
        life_table=life_table_filtered,
        timepoint_initial=life_table_filtered["timepoint"].iloc[0],
        timepoint=timepoint
    )
    assert projected_life_table["timepoint"].nunique() == 1
    assert projected_life_table["timepoint"].iloc[0] == timepoint
    assert projected_life_table["sex"].nunique() == 1
    assert projected_life_table["sex"].iloc[0] == sex
    assert projected_life_table["province"].nunique() == 1
    assert projected_life_table["province"].iloc[0] == province
    assert set(projected_life_table.columns) == set(
        ["province","projection_scenario", "age", "sex", "timepoint", "prob_death"]
    )
    assert set(projected_life_table["age"].unique()) == set(life_table_filtered["age"].unique())
    assert projected_life_table["prob_death"].between(0.0, 1.0).all()



@pytest.mark.parametrize(
    "sex, province, timepoint",
    [
        ("F", "BC", dt.datetime(2026, 1, 1))
    ]
)
def test_get_projected_life_table_single_timepoint_error(life_table, sex, province, timepoint):
    life_table_filtered = life_table.loc[
        (life_table["sex"] == sex) &
        (life_table["province"] == province)
    ].copy()

    with pytest.raises(ValueError, match="Initial life table should only contain one projection scenario."):
        projected_life_table = get_projected_life_table_single_timepoint(
            beta_time=0.1,
            life_table=life_table_filtered,
            timepoint_initial=life_table_filtered["timepoint"].iloc[0],
            timepoint=timepoint
        )


@pytest.mark.parametrize(
    "expected_rows",
    [
        (
            [(
                dt.datetime(2019, 1, 1), "BC", 2, "F",
                0.00015
            )]
        )
    ]
)
def test_load_past_death_data(expected_rows):
    df = load_past_death_data()
    assert df["timepoint"].min() >= MIN_TIMEPOINT
    assert df["sex"].isin(["M", "F"]).all()
    assert set(df.columns) == set([
        "province", "projection_scenario", "age", "sex", "prob_death", "timepoint"
    ])
    assert df["projection_scenario"].isin(["past"]).all()
    assert df["province"].isin(PROVINCE_MAP.values()).all()
    for row in expected_rows:
        assert df.loc[
            (df["timepoint"] == row[0]) &
            (df["province"] == row[1]) &
            (df["age"] == row[2]) &
            (df["sex"] == row[3])
        ].iloc[0]["prob_death"] == pytest.approx(row[4], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "expected_rows",
    [
        [
            ("CA", "F", dt.datetime(2028, 1, 1), "M3", 85.0),
            ("CA", "M", dt.datetime(2028, 1, 1), "M3", 80.9),
            ("CA", "F", dt.datetime(2073, 1, 1), "LG", 89.7),
            ("BC", "M", dt.datetime(2058, 1, 1), "HG", 86.7)
        ]
    ]
)
def test_load_projected_death_data(expected_rows):
    df = load_projected_death_data(min_timepoint=dt.datetime(2022, 1, 1))
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
    for row in expected_rows:
        assert df.loc[
            (df["province"] == row[0]) &
            (df["sex"] == row[1]) &
            (df["timepoint"] == row[2]) &
            (df["projection_scenario"] == row[3])
        ].iloc[0]["life_expectancy"] == row[4]


@pytest.mark.parametrize(
    "beta_parameters, expected_rows",
    [
        (
            {
                ('CA', 'M', 'LG'): -0.014654675868609225,
                ('CA', 'M', 'M3'): -0.014654675868609225,
                ('CA', 'M', 'HG'): -0.014654675868609225,
                ('CA', 'F', 'LG'): -0.014654675868609225,
                ('CA', 'F', 'M3'): -0.014654675868609225,
                ('CA', 'F', 'HG'): -0.014654675868609225,
                ('BC', 'M', 'LG'): -0.014654675868609225,
                ('BC', 'M', 'M3'): -0.014654675868609225,
                ('BC', 'M', 'HG'): -0.014654675868609225,
                ('BC', 'F', 'LG'): -0.014654675868609225,
                ('BC', 'F', 'M3'): -0.014654675868609225,
                ('BC', 'F', 'HG'): -0.014654675868609225

            },
            [
                (dt.datetime(2057, 1, 1), "CA", 60, "F", 0.0027035098555941027),
                (dt.datetime(2042, 1, 1), "BC", 22, "M", 0.0007291262068477189)
            ],
        ),
    ]
)
def test_get_projected_death_data(beta_parameters, expected_rows):
    past_life_table = load_past_death_data()
    df = get_projected_death_data(
        beta_parameters=beta_parameters,
        past_life_table=past_life_table.loc[past_life_table["province"].isin(["BC", "CA"])]
    )
    assert df["timepoint"].min() >= MIN_TIMEPOINT
    assert df["sex"].isin(["M", "F"]).all()
    assert set(df.columns) == set([
        "province", "projection_scenario", "age", "sex", "prob_death", "timepoint"
    ])
    assert df["province"].isin(PROVINCE_MAP.values()).all()
    for row in expected_rows:
        assert df.loc[
            (df["timepoint"] == row[0]) &
            (df["province"] == row[1]) &
            (df["age"] == row[2]) &
            (df["sex"] == row[3])
        ].iloc[0]["prob_death"] == pytest.approx(row[4], rel=0.05)