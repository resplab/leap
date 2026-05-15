import pytest
import datetime as dt
from leap.data_generation.birth_data import get_projection_scenario_id, MIN_TIMEPOINT, \
    CENSUS_TIMEPOINT, load_past_births_population_data, load_projected_births_population_data, \
    load_past_initial_population_data, load_projected_initial_population_data, TIME_DELTA_OD
from leap.logger import get_logger
from leap.utils import TimeDelta

logger = get_logger(__name__)

PROJECTION_SCENARIOS = ["M1", "M2", "M3", "M4", "M5", "M6", "LG", "HG", "FA", "SA"]


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



def test_load_past_births_population_data():
    df = load_past_births_population_data()

    # check that all timepoints are >= MIN_TIMEPOINT and < CENSUS_TIMEPOINT
    assert df["timepoint"].min() >= MIN_TIMEPOINT
    assert df["timepoint"].max() <= CENSUS_TIMEPOINT

    # check that all projection scenarios are "past"
    assert (df["projection_scenario"] == "past").all()

    # check that the expected columns are present
    expected_columns = ["timepoint", "province", "N", "prop_male", "projection_scenario"]
    assert all(column in df.columns for column in expected_columns)

    # check specific values from 17100005.csv
    assert df.loc[
        (df["timepoint"] == dt.datetime(2006, 1, 1)) & 
        (df["province"] == "CA")
    ].iloc[0]["N"] == 344791
    assert df.loc[
        (df["timepoint"] == dt.datetime(2006, 1, 1)) & 
        (df["province"] == "CA")
    ].iloc[0]["prop_male"] == pytest.approx(0.5128556139806434, abs=1e-8)


def test_load_projected_births_population_data():
    df = load_projected_births_population_data(min_timepoint=CENSUS_TIMEPOINT)

    # check that all timepoints are >= CENSUS_TIMEPOINT
    assert df["timepoint"].min() >= CENSUS_TIMEPOINT

    # check that all projection scenarios are in PROJECTION_SCENARIOS
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS).all()

    # check that every projection scenario has the same amount of data and that it's greater than 0
    counts = [
        df.loc[df["projection_scenario"]==projection_scenario].shape[0] 
        for projection_scenario in PROJECTION_SCENARIOS
    ]
    assert len(set(counts)) == 1
    assert counts[0] > 0

    # check that there are no missing values of N
    assert df["N"].isna().sum() == 0

    # check that the expected columns are present
    expected_columns = ["timepoint", "province", "N", "prop_male", "projection_scenario"]
    assert all(column in df.columns for column in expected_columns)

    # check specific values from 17100057.csv
    assert df.loc[
        (df["timepoint"] == dt.datetime(2022, 1, 1)) & 
        (df["province"] == "BC") & 
        (df["projection_scenario"] == "LG")
        ].iloc[0]["N"] == 42300
    assert df.loc[
        (df["timepoint"] == dt.datetime(2022, 1, 1)) & 
        (df["province"] == "BC") & 
        (df["projection_scenario"] == "LG")
        ].iloc[0]["prop_male"] == pytest.approx(0.5130023640661938, abs=1e-8)


@pytest.mark.parametrize(
    "time_delta",
    [
        # (TimeDelta(years=5)),
        (TimeDelta(months=1))
    ]
)
@pytest.mark.parametrize(
    "min_timepoint",
    [
        (MIN_TIMEPOINT)
    ]
)
def test_load_past_initial_population_data(time_delta, min_timepoint):
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
    assert df.loc[
        (df["timepoint"] == dt.datetime(2006, 1, 1)) & 
        (df["province"] == "CA") &
        (df["age"] == 0)
    ].iloc[0]["n_age"] == 344791
    assert df.loc[
        (df["timepoint"] == dt.datetime(2006, 1, 1)) & 
        (df["province"] == "CA") &
        (df["age"] == 0)
    ].iloc[0]["n_birth"] == 344791
    assert df.loc[
        (df["timepoint"] == dt.datetime(2006, 1, 1)) & 
        (df["province"] == "CA") &
        (df["age"] == 0)
    ].iloc[0]["prop_male"] == pytest.approx(0.5128556139806434, abs=1e-8)


@pytest.mark.parametrize(
    "time_delta",
    [
        # (TimeDelta(years=5)),
        (TimeDelta(months=1))
    ]
)
@pytest.mark.parametrize(
    "max_timepoint",
    [
        (dt.datetime(2024, 1, 1)),
    ]
)
def test_load_projected_initial_population_data(time_delta, max_timepoint):
    df = load_projected_initial_population_data(
        time_delta=time_delta, min_timepoint=CENSUS_TIMEPOINT, max_timepoint=max_timepoint
    )

    # check that all timepoints are >= CENSUS_TIMEPOINT
    assert df["timepoint"].min() >= CENSUS_TIMEPOINT

    # check that all projection scenarios are in PROJECTION_SCENARIOS
    assert df["projection_scenario"].isin(PROJECTION_SCENARIOS).all()

    # check that every projection scenario has the same amount of data and that it's greater than 0
    counts = [
        df.loc[df["projection_scenario"]==projection_scenario].shape[0] 
        for projection_scenario in PROJECTION_SCENARIOS
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
    assert df.loc[
        (df["timepoint"] == dt.datetime(2022, 1, 1)) & 
        (df["province"] == "BC") & 
        (df["projection_scenario"] == "LG") &
        (df["age"] == 0)
        ].iloc[0]["n_age"] == 42300
    assert df.loc[
        (df["timepoint"] == dt.datetime(2022, 1, 1)) & 
        (df["province"] == "BC") & 
        (df["projection_scenario"] == "LG") &
        (df["age"] == 0)
        ].iloc[0]["n_birth"] == 42300
    assert df.loc[
        (df["timepoint"] == dt.datetime(2022, 1, 1)) & 
        (df["province"] == "BC") & 
        (df["projection_scenario"] == "LG") &
        (df["age"] == 0)
        ].iloc[0]["prop_male"] == pytest.approx(0.5130023640661938, abs=1e-8)
    


