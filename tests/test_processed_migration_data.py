"""Checks on the generated (processed) migration and population data committed to the repo.

These read the CSVs in ``processed_data/``, so they fail if the data generation code changes
without the data being regenerated.
"""
import pytest
import pandas as pd
from leap.utils import get_data_path, get_time_delta_tag, TimeDelta, PROJECTION_SCENARIOS_FUTURE

TIME_DELTAS = [TimeDelta(years=1), TimeDelta(months=1)]
PROVINCES = ["BC", "CA"]


def load_migration_tables(time_delta: TimeDelta) -> pd.DataFrame:
    folder = get_data_path(f"processed_data/{get_time_delta_tag(time_delta)}/migration")
    return pd.concat(
        [pd.read_csv(file_path, parse_dates=["timepoint"])
         for file_path in sorted(folder.glob("migration_table_*.csv"))],
        ignore_index=True
    )


def load_birth_data(time_delta: TimeDelta, file_name: str) -> pd.DataFrame:
    file_path = get_data_path(f"processed_data/{get_time_delta_tag(time_delta)}/birth/{file_name}")
    return pd.read_csv(file_path, parse_dates=["timepoint"])


@pytest.fixture(scope="module")
def migration_tables() -> dict[str, pd.DataFrame]:
    return {
        time_delta.to_isoformat(): load_migration_tables(time_delta) for time_delta in TIME_DELTAS
    }


@pytest.mark.parametrize("time_delta", TIME_DELTAS)
def test_no_age_below_1(migration_tables, time_delta):
    # age < 1 is handled entirely by the birth model, not migration
    df = migration_tables[time_delta.to_isoformat()]
    assert (df["age"] < 1).sum() == 0


@pytest.mark.parametrize("time_delta", TIME_DELTAS)
def test_immigration_or_emigration_not_both(migration_tables, time_delta):
    # each age / sex / timepoint is net immigration or net emigration, never both
    df = migration_tables[time_delta.to_isoformat()]
    both = (df["prop_immigrants_timepoint"] > 0) & (df["prop_emigrants_timepoint"] > 0)
    assert both.sum() == 0
    assert ((df["prob_emigration"] > 0) & (df["delta_n"] >= 0)).sum() == 0


def check_consecutive(df: pd.DataFrame, time_delta: TimeDelta, label: str):
    """Check that past followed by each projection scenario has no missing timepoints."""
    past = df.loc[df["projection_scenario"] == "past"]
    for (province, projection_scenario), df_projected in df.loc[
        df["projection_scenario"] != "past"
    ].groupby(["province", "projection_scenario"]):
        timepoints = sorted(
            set(past.loc[past["province"] == province, "timepoint"]) |
            set(df_projected["timepoint"])
        )
        expected = [timepoints[0] + time_delta.to_dateoffset() * i for i in range(len(timepoints))]
        assert timepoints == expected, f"{label}: gap in {province} {projection_scenario}"


@pytest.mark.parametrize("time_delta", TIME_DELTAS)
@pytest.mark.parametrize("file_name", ["initial_population.csv", "birth_estimate.csv"])
def test_birth_data_timepoints_consecutive(time_delta, file_name):
    df = load_birth_data(time_delta, file_name)
    check_consecutive(df, time_delta, file_name)


@pytest.mark.parametrize("time_delta", TIME_DELTAS)
def test_migration_timepoints_consecutive(migration_tables, time_delta):
    check_consecutive(migration_tables[time_delta.to_isoformat()], time_delta, "migration")


@pytest.mark.parametrize("time_delta", TIME_DELTAS)
def test_first_projected_timepoint_present(migration_tables, time_delta):
    # the first projected timepoint links to the last past timepoint, so it must not be dropped
    df_migration = migration_tables[time_delta.to_isoformat()]
    df_population = load_birth_data(time_delta, "initial_population.csv")
    for province in PROVINCES:
        for projection_scenario in PROJECTION_SCENARIOS_FUTURE:
            mask_population = (
                (df_population["province"] == province) &
                (df_population["projection_scenario"] == projection_scenario)
            )
            mask_migration = (
                (df_migration["province"] == province) &
                (df_migration["projection_scenario"] == projection_scenario)
            )
            assert (
                df_migration.loc[mask_migration, "timepoint"].min() ==
                df_population.loc[mask_population, "timepoint"].min()
            ), f"{province} {projection_scenario}"


def get_one_year(
    migration_tables: dict[str, pd.DataFrame], province: str, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get the annual step into ``year + 1`` and the 12 monthly steps covering the same year."""
    df_annual = migration_tables[TimeDelta(years=1).to_isoformat()]
    df_monthly = migration_tables[TimeDelta(months=1).to_isoformat()]
    annual = df_annual.loc[
        (df_annual["province"] == province) &
        (df_annual["projection_scenario"] == "M3") &
        (df_annual["timepoint"] == pd.Timestamp(year + 1, 1, 1))
    ]
    monthly = df_monthly.loc[
        (df_monthly["province"] == province) &
        (df_monthly["projection_scenario"] == "M3") &
        (df_monthly["timepoint"] > pd.Timestamp(year, 1, 1)) &
        (df_monthly["timepoint"] <= pd.Timestamp(year + 1, 1, 1))
    ]
    assert monthly["timepoint"].nunique() == 12
    return annual, monthly


@pytest.mark.parametrize("province", PROVINCES)
@pytest.mark.parametrize("year", [2025, 2030, 2040])
def test_monthly_net_migration_matches_annual(migration_tables, province, year):
    # changing the time step shouldn't change net migration over a year
    annual, monthly = get_one_year(migration_tables, province, year)
    assert monthly["delta_n"].sum() == pytest.approx(annual["delta_n"].sum(), rel=0.1)


@pytest.mark.xfail(
    strict=True,
    reason="issue #98: netting migration month by month inflates emigration vs annual"
)
@pytest.mark.parametrize("province", PROVINCES)
def test_monthly_emigration_matches_annual(migration_tables, province):
    annual, monthly = get_one_year(migration_tables, province, 2030)
    emigrants_annual = (annual["prob_emigration"] * annual["n"]).sum()
    emigrants_monthly = (monthly["prob_emigration"] * monthly["n"]).sum()
    assert emigrants_monthly == pytest.approx(emigrants_annual, rel=0.25)
