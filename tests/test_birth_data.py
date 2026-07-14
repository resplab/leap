import pytest
import datetime as dt
from leap.data_generation.birth_data import get_projection_scenario_id, MIN_TIMEPOINT, \
    CENSUS_TIMEPOINT, load_past_births_population_data, load_projected_births_population_data, \
    load_past_initial_population_data, load_projected_initial_population_data, TIME_DELTA_OD
from leap.logger import get_logger
from leap.utils import TimeDelta, PROJECTION_SCENARIOS_FUTURE

logger = get_logger(__name__)


@pytest.mark.parametrize(
    "projection_scenario, expected_id",
    [
        ("Projection scenario M1", "M1"),
        ("Projection scenario LG", "LG"),
        ("Projection scenario HG", "HG"),
    ]
)
def test_get_projection_scenario_id(projection_scenario, expected_id):
    assert get_projection_scenario_id(projection_scenario) == expected_id


@pytest.mark.parametrize(
    "time_delta, expected_rows",
    [
        (
            TimeDelta(years=5, months=6),
            [
                (
                    dt.datetime(2000, 1, 1), "BC",
                    41433 * TimeDelta(years=5, months=6).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    0.514420872251587
                )
            ]
        ),
        (
            TimeDelta(months=1),
            [
                (
                    dt.datetime(2000, 1, 1), "BC",
                    41433 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    0.514420872251587
                ),
                (
                    dt.datetime(2006, 1, 1), "CA",
                    344791 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    0.5128556139806434
                )
            ]
        )
    ]
)
def test_load_past_births_population_data(time_delta, expected_rows):
    df = load_past_births_population_data(time_delta)

    # check that all timepoints are >= MIN_TIMEPOINT and < CENSUS_TIMEPOINT
    assert df["timepoint"].min() >= MIN_TIMEPOINT
    assert df["timepoint"].max() <= CENSUS_TIMEPOINT + TIME_DELTA_OD

    # check that all projection scenarios are "past"
    assert (df["projection_scenario"] == "past").all()

    # check that the expected columns are present
    expected_columns = ["timepoint", "province", "N", "prop_male", "projection_scenario"]
    assert all(column in df.columns for column in expected_columns)

    # check specific values from 17100005.csv
    for row in expected_rows:
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1])
        ].iloc[0]["N"] == row[2]
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1])
        ].iloc[0]["prop_male"] == pytest.approx(row[3], abs=1e-8)


@pytest.mark.parametrize(
    "time_delta, min_timepoint, expected_rows",
    [
        (
            TimeDelta(years=5, months=6),
            dt.datetime(2025, 1, 1),
            [
                (
                    dt.datetime(2025, 1, 1), "CA", "FA", 
                    346300 * TimeDelta(years=5, months=6).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    0.5119838290499567
                )
            ]
        ),
        (
            TimeDelta(months=1),
            CENSUS_TIMEPOINT,
            [
                (
                    dt.datetime(2022, 1, 1), "BC", "LG", 
                    42300 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    0.5130023640661938
                )
            ]
        )
    ]
)
def test_load_projected_births_population_data(time_delta, min_timepoint, expected_rows):
    df = load_projected_births_population_data(min_timepoint=min_timepoint, time_delta=time_delta)

    # check that all timepoints are >= min_timepoint
    assert df["timepoint"].min() >= min_timepoint

    # check that all projection scenarios are in PROJECTION_SCENARIOS
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS_FUTURE).all()

    # check that every projection scenario has the same amount of data and that it's greater than 0
    counts = [
        df.loc[df["projection_scenario"]==projection_scenario].shape[0] 
        for projection_scenario in PROJECTION_SCENARIOS_FUTURE
    ]
    assert len(set(counts)) == 1
    assert counts[0] > 0

    # check that there are no missing values of N
    assert df["N"].isna().sum() == 0

    # check that the expected columns are present
    expected_columns = ["timepoint", "province", "N", "prop_male", "projection_scenario"]
    assert all(column in df.columns for column in expected_columns)

    # check specific values from 17100057.csv
    for row in expected_rows:
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) & 
            (df["projection_scenario"] == row[2])
            ].iloc[0]["N"] == row[3]
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) & 
            (df["projection_scenario"] == row[2])
            ].iloc[0]["prop_male"] == pytest.approx(row[4], abs=1e-8)


@pytest.mark.parametrize(
    "time_delta, min_timepoint, expected_rows",
    [
        (
            TimeDelta(years=5, months=6),
            MIN_TIMEPOINT,
            [(
                dt.datetime(2000, 1, 1), "CA", 0,
                338738 * TimeDelta(years=5, months=6).total_seconds() / TIME_DELTA_OD.total_seconds(),
                338738 * TimeDelta(years=5, months=6).total_seconds() / TIME_DELTA_OD.total_seconds(),
                0.5124137238809936
            )]
        ),
        (
            TimeDelta(months=1),
            MIN_TIMEPOINT,
            [(
                dt.datetime(2006, 1, 1), "CA", 0,
                344791 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds(),
                344791 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds(),
                0.5128556139806434
            )]
        )
    ]
)
def test_load_past_initial_population_data(time_delta, min_timepoint, expected_rows):
    df = load_past_initial_population_data(time_delta=time_delta, min_timepoint=min_timepoint)

    # check that all timepoints are >= MIN_TIMEPOINT and < CENSUS_TIMEPOINT
    assert df["timepoint"].min() >= min_timepoint
    assert df["timepoint"].max() <= CENSUS_TIMEPOINT + TIME_DELTA_OD

    # check that the expected columns are present
    expected_columns = [
        "timepoint", "province", "age", "prop_male", "n_age", "n_birth", "prop", "projection_scenario"
    ]
    assert all(column in df.columns for column in expected_columns)

    # check that all projection scenarios are "past"
    assert (df["projection_scenario"] == "past").all()

    # check that n_birth is the same for all ages for a given timepoint and province
    grouped_df = df.groupby(["timepoint", "province"])
    for (timepoint, province), group in grouped_df:
        assert group["n_birth"].nunique() == 1
        assert group.loc[group["age"] == 0]["n_age"].values[0] == group["n_birth"].values[0]
        assert group.loc[group["age"] == 0]["prop"].values[0] == 1.0

    # check specific values from 17100005.csv
    for row in expected_rows:
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) &
            (df["age"] == row[2])
        ].iloc[0]["n_age"] == row[3]
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) &
            (df["age"] == row[2])
        ].iloc[0]["n_birth"] == row[4]
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) &
            (df["age"] == row[2])
        ].iloc[0]["prop_male"] == pytest.approx(row[5], abs=1e-8)


@pytest.mark.parametrize(
    "time_delta, min_timepoint, max_timepoint, expected_rows",
    [
        (
            TimeDelta(years=5, months=6),
            dt.datetime(2025, 1, 1),
            dt.datetime(2030, 1, 1),
            [
                (
                    dt.datetime(2025, 1, 1), "CA", "FA", 0,
                    346300 * TimeDelta(years=5, months=6).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    346300 * TimeDelta(years=5, months=6).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    0.5119838290499567
                )
            ]
        ),
        (
            TimeDelta(months=1),
            dt.datetime(2022, 1, 1),
            dt.datetime(2024, 1, 1),
            [
                (
                    dt.datetime(2022, 1, 1), "BC", "LG", 0,
                    42300 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    42300 * TimeDelta(months=1).total_seconds() / TIME_DELTA_OD.total_seconds(),
                    0.5130023640661938
                )
            ]
        )
    ]
)
def test_load_projected_initial_population_data(
    time_delta, min_timepoint, max_timepoint, expected_rows
):
    df = load_projected_initial_population_data(
        time_delta=time_delta, min_timepoint=min_timepoint, max_timepoint=max_timepoint
    )

    # check that all timepoints are >= min_timepoint
    assert df["timepoint"].min() >= min_timepoint

    # check that all projection scenarios are in PROJECTION_SCENARIOS
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS_FUTURE).all()

    # check that every projection scenario has the same amount of data and that it's greater than 0
    counts = [
        df.loc[df["projection_scenario"]==projection_scenario].shape[0] 
        for projection_scenario in PROJECTION_SCENARIOS_FUTURE
    ]
    assert len(set(counts)) == 1
    assert counts[0] > 0

    # check that there are no missing values of N
    assert df["n_age"].isna().sum() == 0

    # check that the expected columns are present
    expected_columns = [
        "timepoint", "province", "age", "prop_male", "n_age", "n_birth", "prop", "projection_scenario"
    ]
    assert all(column in df.columns for column in expected_columns)

    # check that n_birth is the same for all ages for a given timepoint, province, and projection scenario
    grouped_df = df.groupby(["timepoint", "province", "projection_scenario"])
    for (timepoint, province, projection_scenario), group in grouped_df:
        assert group["n_birth"].nunique() == 1
        assert group.loc[group["age"] == 0]["n_age"].values[0] == group["n_birth"].values[0]
        assert group.loc[group["age"] == 0]["prop"].values[0] == 1.0

    # check specific values from 17100057.csv
    for row in expected_rows:
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) & 
            (df["projection_scenario"] == row[2]) &
            (df["age"] == row[3])
            ].iloc[0]["n_age"] == row[4]
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) & 
            (df["projection_scenario"] == row[2]) &
            (df["age"] == row[3])
            ].iloc[0]["n_birth"] == row[5]
        assert df.loc[
            (df["timepoint"] == row[0]) & 
            (df["province"] == row[1]) & 
            (df["projection_scenario"] == row[2]) &
            (df["age"] == row[3])
            ].iloc[0]["prop_male"] == pytest.approx(row[6], abs=1e-8)
    


