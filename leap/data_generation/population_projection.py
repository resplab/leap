"""Validate the birth, death and migration data by projecting the population forward.

This is a deterministic, aggregate version of the demographic part of the simulation. It uses the
same module classes (``Birth``, ``Death``, ``Emigration``, ``Immigration``) and processed data as
``Simulation``, but tracks expected counts by age and sex instead of individual agents, so it runs
in seconds and has no random noise.

Each time step mirrors ``Simulation.get_new_agents`` and ``Simulation.simulate_agent``:

1. At the first timepoint, the initial population is created from ``Birth.initial_population``.
   At later timepoints, newborns (``Birth.get_num_newborn``) and immigrants
   (``Immigration.get_num_new_immigrants``, distributed by ``prop_immigrants_timepoint``) are added.
2. People die with the life table probability for their age, sex and timepoint.
3. Survivors emigrate with ``prob_emigration`` (never at age 0, as in ``Emigration``).
4. The rest are counted as alive, and move on one time step.

Ageing follows the intended model: people age by one ``time_delta`` per step, and the integer age
used for table lookups is their age in whole years. For annual steps this is the same as
``Simulation``. For sub-annual steps, each integer age is split evenly across its sub-ages, as in
the data generation.

The results have the same shape as the simulation's ``outcome_matrix_alive.csv``,
``outcome_matrix_death.csv``, ``outcome_matrix_immigration.csv`` and
``outcome_matrix_emigration.csv``, and can be compared with StatCan's projected population in
``initial_population.csv``.
"""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
from dataclasses import dataclass
from leap.birth import Birth
from leap.death import Death
from leap.emigration import Emigration
from leap.immigration import Immigration
from leap.utils import TimeDelta, date_range, get_data_path, get_time_delta_tag
from leap.logger import get_logger

logger = get_logger(__name__, 20)

SEXES = ["F", "M"]
MAX_AGE = 110  # oldest age in the life tables
MAX_AGE_EMIGRATION = 100  # oldest age in the migration tables; Emigration uses min(age, 100)


@dataclass
class PopulationProjection:
    """The results of a population projection, one row per timepoint, age and sex.

    Attributes:
        alive: ``timepoint, age, sex, n_alive``: people alive at the end of each step, as in
            ``outcome_matrix_alive.csv``.
        death: ``timepoint, age, sex, n_deaths``.
        immigration: ``timepoint, age, sex, n_immigrants``.
        emigration: ``timepoint, age, sex, n_emigrants``.
        births: ``timepoint, n_births``.
        population: ``timepoint, age, sex, n``: people at the start of each step, after newborns
            and immigrants are added. This is the number to compare with StatCan's population.
    """
    alive: pd.DataFrame
    death: pd.DataFrame
    immigration: pd.DataFrame
    emigration: pd.DataFrame
    births: pd.DataFrame
    population: pd.DataFrame


def get_max_time_horizon(
    province: str, projection_scenario: str, min_timepoint: dt.datetime
) -> TimeDelta:
    """Get the longest time horizon covered by both the annual and monthly migration data.

    Returns:
        A time horizon in whole years, so annual and monthly projections cover the same years.
    """
    last_years = []
    for time_delta in [TimeDelta(years=1), TimeDelta(months=1)]:
        file_path = get_data_path(
            f"processed_data/{get_time_delta_tag(time_delta)}/migration/"
            f"migration_table_{province}_{projection_scenario}.csv"
        )
        timepoints = pd.read_csv(file_path, usecols=["timepoint"], parse_dates=["timepoint"])
        last_years.append(timepoints["timepoint"].max().year)
    return TimeDelta(years=min(last_years) - min_timepoint.year + 1)


def project_population(
    province: str = "BC",
    projection_scenario: str = "M3",
    min_timepoint: dt.datetime = dt.datetime(2025, 1, 1),
    time_horizon: TimeDelta | None = None,
    time_delta: TimeDelta = TimeDelta(years=1),
    num_births_initial: float | None = None,
) -> PopulationProjection:
    """Project the population forward using the birth, death and migration modules.

    Args:
        province: The 2-letter province ID, e.g. ``BC``.
        projection_scenario: The projection scenario, e.g. ``M3``.
        min_timepoint: The first timepoint.
        time_horizon: How far to project. If ``None``, the full range of the migration data
            (see ``get_max_time_horizon``).
        time_delta: The time step.
        num_births_initial: The number of births per year at ``min_timepoint``, as in the
            simulation config. If ``None``, StatCan's births for that year are used, so the results
            are on the same scale as StatCan's population.

    Returns:
        The projection results.
    """
    if time_horizon is None:
        time_horizon = get_max_time_horizon(province, projection_scenario, min_timepoint)
    n_intervals = TimeDelta(years=1) // time_delta
    birth = Birth(min_timepoint, province, projection_scenario, MAX_AGE, time_delta=time_delta)
    death = Death(province, projection_scenario, min_timepoint, time_delta=time_delta)
    emigration = Emigration(min_timepoint, province, projection_scenario, time_delta=time_delta)
    immigration = Immigration(
        min_timepoint, province, projection_scenario, MAX_AGE, time_delta=time_delta
    )
    if num_births_initial is None:
        num_births_initial = float(birth.estimate.get_group(min_timepoint)["N"].iloc[0])

    # population by sex and sub-age; sub-age k is age k / n_intervals years
    n_sub_ages = (MAX_AGE + 1) * n_intervals
    integer_age = np.arange(n_sub_ages) // n_intervals
    population = {sex: np.zeros(n_sub_ages) for sex in SEXES}

    def add_by_age(sex: str, ages: np.ndarray, counts: np.ndarray):
        """Add counts at integer ages, split evenly across each age's sub-ages."""
        for age, count in zip(ages, counts):
            age = min(int(age), MAX_AGE)
            population[sex][age * n_intervals:(age + 1) * n_intervals] += count / n_intervals

    def by_integer_age(counts: np.ndarray) -> np.ndarray:
        return np.bincount(integer_age, weights=counts, minlength=MAX_AGE + 1)

    records = {name: [] for name in ["alive", "death", "immigration", "emigration", "population"]}
    births = []

    def record(name: str, timepoint: dt.datetime, sex: str, counts_by_age: np.ndarray):
        records[name].append(pd.DataFrame({
            "timepoint": timepoint, "age": np.arange(MAX_AGE + 1), "sex": sex,
            "value": counts_by_age
        }))

    timepoints = list(date_range(min_timepoint, min_timepoint + time_horizon, time_delta))
    for timepoint in timepoints:
        immigrants = {sex: np.zeros(MAX_AGE + 1) for sex in SEXES}
        if timepoint == min_timepoint:
            df = birth.initial_population
            for sex, prop_sex in [("F", 1 - df["prop_male"]), ("M", df["prop_male"])]:
                add_by_age(sex, df["age"].to_numpy(), (df["prop"] * num_births_initial * prop_sex).to_numpy())
        else:
            num_new_born = birth.get_num_newborn(num_births_initial, timepoint)
            prop_male = birth.estimate.get_group(timepoint)["prop_male"].iloc[0]
            population["F"][0] += num_new_born * (1 - prop_male)
            population["M"][0] += num_new_born * prop_male
            births.append({"timepoint": timepoint, "n_births": num_new_born})

            num_immigrants = immigration.get_num_new_immigrants(num_new_born, timepoint)
            df = immigration.get_table_group(timepoint)
            for sex in SEXES:
                df_sex = df.loc[df["sex"] == sex]
                counts = num_immigrants * df_sex["prop_immigrants_timepoint"].to_numpy()
                add_by_age(sex, df_sex["age"].to_numpy(), counts)
                np.add.at(immigrants[sex], np.minimum(df_sex["age"].to_numpy(), MAX_AGE), counts)

        df_death = death.life_table.get_group(timepoint).set_index("age")
        df_emigration = emigration.table.get_group(timepoint)
        for sex in SEXES:
            record("population", timepoint, sex, by_integer_age(population[sex]))
            record("immigration", timepoint, sex, immigrants[sex])

            prob_death = df_death[sex].reindex(integer_age).to_numpy()
            deaths = population[sex] * prob_death

            prob_emigration = (
                df_emigration.loc[df_emigration["sex"] == sex]
                .set_index("age")["prob_emigration"]
                .reindex(np.minimum(integer_age, MAX_AGE_EMIGRATION))
                .fillna(0.0).to_numpy(copy=True)
            )
            prob_emigration[integer_age == 0] = 0.0
            emigrants = (population[sex] - deaths) * prob_emigration

            population[sex] = population[sex] - deaths - emigrants
            record("death", timepoint, sex, by_integer_age(deaths))
            record("emigration", timepoint, sex, by_integer_age(emigrants))
            record("alive", timepoint, sex, by_integer_age(population[sex]))

            # move on one time step; people past the oldest age leave the population
            population[sex] = np.concatenate([[0.0], population[sex][:-1]])

    def combine(name: str, column: str) -> pd.DataFrame:
        return pd.concat(records[name], ignore_index=True).rename(columns={"value": column})

    return PopulationProjection(
        alive=combine("alive", "n_alive"),
        death=combine("death", "n_deaths"),
        immigration=combine("immigration", "n_immigrants"),
        emigration=combine("emigration", "n_emigrants"),
        births=pd.DataFrame(births, columns=["timepoint", "n_births"]),
        population=combine("population", "n"),
    )


def summarize_by_year(projection: PopulationProjection) -> pd.DataFrame:
    """Summarize a projection by calendar year.

    Population is taken at the first timepoint of each year, and flows (births, deaths, immigrants,
    emigrants) are summed over the year, so annual and sub-annual projections can be compared.

    Returns:
        A dataframe with columns ``year, population, births, deaths, immigrants, emigrants``.
    """
    def yearly_sum(df: pd.DataFrame, column: str) -> pd.Series:
        return df.groupby(df["timepoint"].dt.year)[column].sum()

    population = projection.population
    first = population.groupby(population["timepoint"].dt.year)["timepoint"].transform("min")
    summary = pd.DataFrame({
        "population": population.loc[population["timepoint"] == first]
        .groupby(population["timepoint"].dt.year)["n"].sum(),
        "births": yearly_sum(projection.births, "n_births"),
        "deaths": yearly_sum(projection.death, "n_deaths"),
        "immigrants": yearly_sum(projection.immigration, "n_immigrants"),
        "emigrants": yearly_sum(projection.emigration, "n_emigrants"),
    })
    summary.index.name = "year"
    return summary.reset_index()


def load_statcan_population(
    province: str, projection_scenario: str, time_delta: TimeDelta = TimeDelta(years=1)
) -> pd.DataFrame:
    """Load StatCan's population (past estimates and projections) from ``initial_population.csv``.

    Returns:
        A dataframe with columns ``timepoint, age, n``.
    """
    file_path = get_data_path(
        f"processed_data/{get_time_delta_tag(time_delta)}/birth/initial_population.csv"
    )
    df = pd.read_csv(file_path, parse_dates=["timepoint"])
    df = df.loc[
        (df["province"] == province) &
        (df["projection_scenario"].isin(["past", projection_scenario]))
    ]
    return df[["timepoint", "age", "n_age"]].rename(columns={"n_age": "n"})


def compare_with_statcan(
    projection: PopulationProjection, province: str, projection_scenario: str,
    age_groups: list[tuple[int, int]] = [(0, 19), (20, 39), (40, 64), (65, MAX_AGE)]
) -> pd.DataFrame:
    """Compare the projected population at the start of each year with StatCan's.

    Returns:
        A dataframe with columns ``year, age_group, projected, statcan, ratio``, including an
        ``"all"`` age group.
    """
    statcan = load_statcan_population(province, projection_scenario)
    statcan = statcan.loc[statcan["timepoint"].dt.month == 1]
    population = projection.population.loc[projection.population["timepoint"].dt.month == 1]
    population = population.groupby(["timepoint", "age"], as_index=False)["n"].sum()

    rows = []
    for label, (low, high) in [("all", (0, MAX_AGE))] + [(f"{a}-{b}", (a, b)) for a, b in age_groups]:
        for df, column in [(population, "projected"), (statcan, "statcan")]:
            df_group = df.loc[(df["age"] >= low) & (df["age"] <= high)]
            rows.append(
                df_group.groupby(df_group["timepoint"].dt.year)["n"].sum()
                .rename(column).to_frame().assign(age_group=label)
            )
    df = pd.concat(rows).reset_index().rename(columns={"timepoint": "year"})
    df = df.groupby(["year", "age_group"], as_index=False).first()
    df = df.dropna(subset=["projected", "statcan"])
    df["ratio"] = df["projected"] / df["statcan"]
    return df


# the same style in every figure; distinct dashes keep series visible where they overlap
STYLE = {
    "color_discrete_map": {"StatCan": "#1f77b4", "annual": "#d62728", "monthly": "#2ca02c"},
    "line_dash_map": {"StatCan": "solid", "annual": "dash", "monthly": "dot"},
    "category_orders": {"projection": ["StatCan", "annual", "monthly"]},
}


def apply_style(fig):
    """Draw StatCan thicker, so the projections are visible on top of it."""
    fig.for_each_trace(
        lambda trace: trace.update(line_width=5 if trace.name == "StatCan" else 2.5)
    )


def plot_comparison(
    summaries: dict[str, pd.DataFrame], statcan: pd.DataFrame, file_path, title: str = ""
):
    """Plot yearly population and flows for several projections, with StatCan's population."""
    frames = [statcan.assign(series="population", projection="StatCan")]
    for label, summary in summaries.items():
        frames.append(summary.melt(id_vars="year", var_name="series").assign(projection=label))
    df = pd.concat(frames, ignore_index=True)
    fig = px.line(
        df, x="year", y="value", color="projection", line_dash="projection",
        facet_col="series", facet_col_wrap=3, markers=True, title=title, **STYLE
    )
    apply_style(fig)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.write_image(str(file_path), scale=2, width=1800, height=1000)


def plot_age_distribution(
    projections: dict[str, PopulationProjection], statcan: pd.DataFrame, years: list[int],
    file_path, title: str = ""
):
    """Plot the population by single year of age for some years, for each projection and StatCan.

    Args:
        projections: The projections to plot, keyed by label.
        statcan: StatCan's population, as returned by ``load_statcan_population``.
        years: The years to plot; the population at the first timepoint of each year is used.
    """
    frames = []
    # StatCan first, so projections that match it closely are drawn on top of it
    sources = {"StatCan": statcan}
    sources.update({label: projection.population for label, projection in projections.items()})
    for label, df in sources.items():
        df = df.loc[(df["timepoint"].dt.month == 1) & (df["timepoint"].dt.year.isin(years))]
        df = df.groupby([df["timepoint"].dt.year.rename("year"), "age"], as_index=False)["n"].sum()
        frames.append(df.assign(projection=label))
    fig = px.line(
        pd.concat(frames, ignore_index=True), x="age", y="n", color="projection",
        line_dash="projection", facet_col="year", title=title, labels={"n": "population"},
        **STYLE
    )
    apply_style(fig)
    fig.write_image(str(file_path), scale=2, width=1800, height=600)


def main(
    province: str = "BC",
    projection_scenario: str = "M3",
    min_timepoint: dt.datetime = dt.datetime(2025, 1, 1),
    time_horizon: TimeDelta | None = None,
):
    """Project the population with annual and monthly steps, compare, and save figures."""
    summaries = {}
    projections = {}
    for label, time_delta in [("annual", TimeDelta(years=1)), ("monthly", TimeDelta(months=1))]:
        projection = project_population(
            province, projection_scenario, min_timepoint, time_horizon, time_delta
        )
        projections[label] = projection
        summaries[label] = summarize_by_year(projection)
        logger.info(f"{label}:\n{summaries[label].round(0).to_string(index=False)}")
        comparison = compare_with_statcan(projection, province, projection_scenario)
        logger.info(
            f"{label} vs StatCan (projected / StatCan):\n"
            f"{comparison.pivot(index='year', columns='age_group', values='ratio').round(3)}"
        )

    statcan = compare_with_statcan(projection, province, projection_scenario)
    statcan = statcan.loc[statcan["age_group"] == "all", ["year", "statcan"]]
    file_path = get_data_path(
        f"data_generation/figures/population_projection_{province}_{projection_scenario}.png",
        mkdirs=True
    )
    plot_comparison(
        summaries, statcan.rename(columns={"statcan": "value"}), file_path,
        title=f"Population projection, {province} {projection_scenario}"
    )
    logger.info(f"Saved figure to {file_path}")

    years = sorted(summaries["annual"]["year"])
    file_path = get_data_path(
        f"data_generation/figures/population_projection_age_{province}_{projection_scenario}.png",
        mkdirs=True
    )
    plot_age_distribution(
        projections, load_statcan_population(province, projection_scenario),
        [years[0], years[len(years) // 2], years[-1]], file_path,
        title=f"Population by age, {province} {projection_scenario}"
    )
    logger.info(f"Saved figure to {file_path}")


if __name__ == "__main__":
    main()
