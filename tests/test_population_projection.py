"""Validation tests: does the generated birth, death and migration data reproduce the population?

Migration is a calibration term (the population change not explained by births and deaths), so it
isn't tested directly; errors in it show up in the population checks.

These run the population projection in ``leap/data_generation/population_projection.py`` and check
the results against StatCan's projected population, and annual against monthly time steps. They
are not run in CI. Run them after regenerating any birth, death or migration data:

    pytest -m validation tests/test_population_projection.py

``test_projection_matches_simulation`` runs a short annual simulation (about 1–2 minutes), to check
that the projection reproduces what ``Simulation`` does.
"""
import pytest
import datetime as dt
from leap.data_generation.population_projection import project_population, \
    summarize_by_year, compare_with_statcan, load_statcan_population
from leap.simulation import Simulation
from leap.utils import TimeDelta

pytestmark = pytest.mark.validation

PROVINCE = "BC"
PROJECTION_SCENARIO = "M3"
MIN_TIMEPOINT = dt.datetime(2025, 1, 1)
TIME_HORIZON = None  # the full range of the migration data
TIME_DELTAS = {"annual": TimeDelta(years=1), "monthly": TimeDelta(months=1)}


@pytest.fixture(scope="module")
def projections():
    return {
        label: project_population(
            PROVINCE, PROJECTION_SCENARIO, MIN_TIMEPOINT, TIME_HORIZON, time_delta
        )
        for label, time_delta in TIME_DELTAS.items()
    }


@pytest.fixture(scope="module")
def summaries(projections):
    summaries = {label: summarize_by_year(projection) for label, projection in projections.items()}
    # the first year has no births or immigrants in the annual projection (initial population only)
    return {label: summary.loc[summary["year"] > MIN_TIMEPOINT.year] for label, summary in summaries.items()}


@pytest.mark.parametrize("label", TIME_DELTAS)
def test_population_matches_statcan(projections, label):
    # births, deaths and migration are calibrated to StatCan's projection, so the projected
    # population should reproduce it
    comparison = compare_with_statcan(projections[label], PROVINCE, PROJECTION_SCENARIO)
    total = comparison.loc[comparison["age_group"] == "all"]
    assert total["ratio"].between(0.99, 1.01).all(), total.to_string()


@pytest.mark.parametrize("label", TIME_DELTAS)
def test_age_groups_match_statcan(projections, label):
    comparison = compare_with_statcan(projections[label], PROVINCE, PROJECTION_SCENARIO)
    assert comparison["ratio"].between(0.98, 1.02).all(), comparison.to_string()


@pytest.mark.parametrize("label", TIME_DELTAS)
def test_single_ages_match_statcan(projections, label):
    # broad age groups can hide large errors at single ages that cancel out; ages 0 (births) and
    # 96+ (small numbers) are excluded
    statcan = load_statcan_population(PROVINCE, PROJECTION_SCENARIO)
    statcan = statcan.loc[statcan["timepoint"].dt.month == 1].set_index(["timepoint", "age"])["n"]
    population = projections[label].population
    population = population.loc[population["timepoint"].dt.month == 1]
    population = population.groupby(["timepoint", "age"])["n"].sum()
    ratio = (population / statcan).dropna()
    ratio = ratio.loc[ratio.index.get_level_values("age").isin(range(1, 96))]
    outside = ratio.loc[~ratio.between(0.9, 1.1)]
    assert outside.empty, outside.round(3).to_string()


def test_monthly_births_match_annual(summaries):
    annual = summaries["annual"].set_index("year")["births"]
    monthly = summaries["monthly"].set_index("year")["births"]
    assert (monthly / annual).between(0.98, 1.02).all(), (monthly / annual).to_string()


def test_monthly_population_matches_annual(summaries):
    annual = summaries["annual"].set_index("year")["population"]
    monthly = summaries["monthly"].set_index("year")["population"]
    assert (monthly / annual).between(0.99, 1.01).all(), (monthly / annual).to_string()


def test_monthly_deaths_match_annual(summaries):
    # monthly deaths are about 4% higher than annual, since people move to the next age's (higher)
    # death rate during the year instead of at the end of it
    annual = summaries["annual"].set_index("year")["deaths"]
    monthly = summaries["monthly"].set_index("year")["deaths"]
    assert (monthly / annual).between(1.02, 1.07).all(), (monthly / annual).to_string()


def test_projection_matches_simulation():
    # the projection should reproduce the simulation's population; checked with annual steps,
    # since the simulation doesn't yet support sub-annual steps
    num_births_initial = 30
    time_horizon = TimeDelta(years=3)
    simulation = Simulation(
        time_delta=TimeDelta(years=1), province=PROVINCE, projection_scenario=PROJECTION_SCENARIO,
        min_timepoint=MIN_TIMEPOINT, time_horizon=time_horizon,
        num_births_initial=num_births_initial
    )
    outcome_matrix = simulation.run(seed=1, until_all_die=False)
    projection = project_population(
        PROVINCE, PROJECTION_SCENARIO, MIN_TIMEPOINT, time_horizon, TimeDelta(years=1),
        num_births_initial=num_births_initial
    )

    alive = outcome_matrix.alive.data
    simulated = alive.groupby(alive["timepoint"].astype("datetime64[ns]"))["n_alive"].sum()
    projected = projection.alive.groupby("timepoint")["n_alive"].sum()
    ratio = simulated / projected.reindex(simulated.index)
    # the simulation is random, so allow for noise (about 3,700 people here)
    assert ratio.between(0.98, 1.02).all(), ratio.to_string()

    immigrants_simulated = outcome_matrix.immigration.data["n_immigrants"].sum()
    immigrants_projected = projection.immigration["n_immigrants"].sum()
    assert immigrants_simulated == pytest.approx(immigrants_projected, rel=0.05)
