import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import pathlib
from leap.utils import get_data_path, get_time_delta_tag, TimeDelta, DATE_FORMAT
from leap.logger import get_logger
from leap.data_generation.utils import get_parser, split_ages
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_TIMEPOINT = dt.datetime(2000, 1, 1)
MIN_TIMEPOINT_PROJ = dt.datetime(2021, 1, 1)
MAX_TIMEPOINT = dt.datetime(2065, 1, 1)
PROVINCES = ["CA", "BC"]
TIME_DELTA_OD = TimeDelta(years=1) # migration data is generated at annual resolution


def get_delta_n(n: float, n_prev: float, prob_death: float) -> float:
    """Get the population change due to migration for a given age and sex in a single time interval.

    Args:
        n: The number of people living in Canada for a single age, sex, timepoint, province, and
            projection scenario.
        n_prev: The number of people living in Canada in the previous time interval for the same
            age, sex, province, and projection scenario as defined for ``n``.
            So if ``n`` is the number of females aged ``10`` in the year
            ``2020``, ``n_prev`` is the number of females aged ``9`` in the year ``2019``.
        prob_death: The probability that a person with a given age and sex at a given
            timepoint will die between the previous timepoint and this timepoint. So if the person is
            a female aged ``10`` in ``2020``, ``prob_death`` is the probability that a
            female aged ``9`` in ``2019`` will die by the age of ``10``.

    Returns:
        The change in population for a given timepoint, age, and sex due to migration.
    """
    return n - n_prev * (1 - prob_death)

def load_population_data(time_delta: TimeDelta) -> pd.DataFrame:
    """Load the population data for the given time delta.

    Args:
        time_delta: The duration of time between subsequent data points.
    
    Returns:
        A dataframe with the following columns:

        * ``timepoint``: The timepoint which the data applies to.
        * ``province``: A string indicating the 2-letter province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``age``: The integer age.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``n``: The number of people living in Canada for a single age, sex, timepoint, province,
          and projection scenario.
        * ``projection_scenario``: The projection scenario.
    """
    logger.info("Loading initial population data from CSV file...")
    time_delta_tag = get_time_delta_tag(time_delta)
    df = pd.read_csv(
        get_data_path(f"processed_data/{time_delta_tag}/birth/initial_population.csv"),
        parse_dates=["timepoint"]
    )

    # Select only the data for timepoints after the min timepoint
    df = df.loc[df["timepoint"] >= MIN_TIMEPOINT]

    df = df[["timepoint", "age", "province", "n_age", "prop_male", "projection_scenario"]]

    # Get the total number of M / F for each year, age, and projection scenario
    df["prop_female"] = 1 - df["prop_male"]
    df.rename(columns={"prop_female": "F", "prop_male": "M"}, inplace=True)
    df = df.melt(
        id_vars=["timepoint", "age", "province", "projection_scenario", "n_age"],
        value_vars=["M", "F"],
        var_name="sex",
        value_name="prop"
    )
    df["n"] = (df["n_age"] * df["prop"]).astype(int)
    df.drop(columns=["n_age", "prop"], inplace=True)

    if time_delta < TimeDelta(years=1):
        df = split_ages(df, time_delta, TimeDelta(years=1), ["n"])

    return df


def load_migration_data(time_delta: TimeDelta) -> pd.DataFrame:
    """Generate migration data for the given provinces and years.

    Args:
        time_delta: The duration of time between subsequent data points.

    Returns:
        A dataframe with the following columns:

        * ``timepoint``: The timepoint which the data applies to.
        * ``province``: A string indicating the 2-letter province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``projection_scenario``: The projection scenario.
        * ``delta_n``: The signed change in population for a given timepoint, age, sex, province, and
          projection scenario due to net migration. Positive values indicate net immigration;
          negative values indicate net emigration.
        * ``prop_migrants_birth``: The signed proportion of ``delta_n`` relative to the total
          number of births in that time interval for the given province and projection scenario.
          Positive = net immigration, negative = net emigration.
        * ``prop_immigrants_timepoint``: For cells where ``delta_n > 0``, the proportion of immigrants
          for this age and sex relative to the total number of immigrants in that time interval.
          Zero for emigration cells. Denominator includes only immigration cells.
        * ``prop_emigrants_timepoint``: For cells where ``delta_n < 0``, the proportion of emigrants
          for this age and sex relative to the total number of emigrants in that time interval.
          Zero for immigration cells. Denominator includes only emigration cells.
        * ``prob_emigration``: For cells where ``delta_n < 0``, the per-person probability of
          emigrating (``abs(delta_n) / N``). Zero for immigration cells.

    """

    logger.info("Loading mortality data from CSV file...")
    time_delta_tag = get_time_delta_tag(time_delta)

    life_table = pd.DataFrame({
        "province": [],
        "projection_scenario": [],
        "timepoint": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "prob_death": np.array([], dtype=float)
    })
    for file_path in get_data_path(f"processed_data/{time_delta_tag}/death").glob("life_table_*.csv"):
        df = pd.read_csv(get_data_path(file_path), parse_dates=["timepoint"])
        life_table = pd.concat([life_table, df], axis=0)

    if time_delta < TimeDelta(years=1):
        life_table = split_ages(life_table, time_delta, TimeDelta(years=1), [])

    # Load the population data from the CSV file
    df_population = load_population_data(time_delta)

    # Select only the data for the given province
    df = df_population.copy()

    # join to the life table to get death probabilities
    df = df.merge(
        life_table, on=["timepoint", "age", "province", "sex"], how="left"
    )

    # get the number of births in each year
    df_birth = df.loc[df["age"] == 0]
    grouped_df = df_birth.groupby(["timepoint", "province", "projection_scenario"])
    df_birth["n_birth"] = grouped_df.transform("sum")["n"]
    df_birth = df_birth.loc[df_birth["sex"] == "F", ["timepoint", "n_birth", "projection_scenario", "province"]]

    # get the previous timepoint's cohort for each entry
    df["age_key"] = df["age"] - time_delta.total_years()
    df["timepoint_key"] = df["timepoint"].apply(
        lambda x: x - time_delta
    )

    df_prev = df[
        ["province", "projection_scenario", "sex", "age", "timepoint", "n", "prob_death"]
    ].rename(columns={
        "age": "age_key",
        "timepoint": "timepoint_key",
        "n": "n_prev",
        "prob_death": "prob_death_prev"
    })

    df = df.merge(
        df_prev,
        on=["province", "projection_scenario", "sex", "age_key", "timepoint_key"],
        how="left"
    )
    df["age_prev"] = df["age_key"]
    df["timepoint_prev"] = df["timepoint_key"]
    df = df.drop(columns=["age_key", "timepoint_key"])

    # remove the missing data
    df = df.dropna(subset=["n_prev"])

    # compute the signed population change due to net migration
    df["delta_n"] = df["n"] - df["n_prev"] * (1 - df["prob_death_prev"])

    # number of migrants
    df["n_immigrants"] = df["delta_n"].clip(lower=0)
    df["n_emigrants"] = (-df["delta_n"]).clip(lower=0)

    # add the n_birth column to df
    df = pd.merge(
        df, df_birth, on=["province", "projection_scenario", "timepoint"], how="left"
    )

    # signed proportion relative to births
    df["prop_migrants_birth"] = df["delta_n"] / df["n_birth"]

    # timepoint proportions with separate denominators for immigration and emigration
    df["n_immigrants_timepoint"] = df.groupby("timepoint")["n_immigrants"].transform("sum")
    df["n_emigrants_timepoint"] = df.groupby("timepoint")["n_emigrants"].transform("sum")
    df["prop_immigrants_timepoint"] = df["n_immigrants"] / df["n_immigrants_timepoint"]
    df["prop_emigrants_timepoint"] = df["n_emigrants"] / df["n_emigrants_timepoint"]
    df = df.fillna(0)

    # per-person probability of emigrating
    df["prob_emigration"] = df["n_emigrants"] / df["n"]

    df = df.drop(columns=["n_prev", "prob_death_prev", "age_prev", "timepoint_prev"])

    return df


def generate_migration_data(time_delta: TimeDelta):
    df_migration = load_migration_data(time_delta)
    time_delta_tag = get_time_delta_tag(time_delta)

    # Convert the age back to integer
    df_migration["age_int"] = df_migration["age"].apply(lambda x: int(x))
    df_migration = df_migration.groupby(
        ["age_int", "province", "sex", "timepoint", "projection_scenario"],
        as_index=False
    ).agg({
        "delta_n": "sum",
        "prop_migrants_birth": "mean",
        "prop_immigrants_timepoint": "mean",
        "prop_emigrants_timepoint": "mean",
        "prob_emigration": "mean",
        "n_immigrants": "sum",
        "n_emigrants": "sum"
    })
    df_migration.rename(columns={"age_int": "age"}, inplace=True)
    df_migration.sort_values(["province", "sex", "timepoint", "age"], inplace=True)

    # Create validation plots for the migration data
    timepoints = df_migration.loc[df_migration["timepoint"] > MIN_TIMEPOINT_PROJ, "timepoint"].unique()
    indices = np.arange(start=0, stop=len(timepoints), step=max(1, len(timepoints) // 4))
    timepoints = timepoints[indices]

    df = df_migration.loc[df_migration["timepoint"].isin(timepoints)].copy()
    df = df.melt(
        id_vars=["age", "sex", "province", "timepoint", "projection_scenario"],
        value_vars=[
            "n_immigrants",
            "n_emigrants",
            "delta_n",
            "prop_immigrants_timepoint",
            "prop_emigrants_timepoint",
            "prob_emigration"
        ],
        var_name="series",
        value_name="value"
    )
    for province in df["province"].unique():
        for sex in df["sex"].unique():
            plot(
                df=df.loc[
                    (df["series"].isin(["n_immigrants", "n_emigrants", "delta_n"])) &
                    (df["province"] == province) &
                    (df["sex"] == sex)
                ],
                y="value",
                ylabel="N",
                color="series",
                title=f"Net Migration, Sex = {sex}, Province = {province}",
                file_path=get_data_path(
                    f"data_generation/figures/{time_delta_tag}/migration/delta_n_{province}_{sex}.png",
                    mkdirs=True
                ),
                height=5000,
                width=3000
            )
            plot(
                df=df.loc[
                    (df["series"].isin(
                        ["prop_immigrants_timepoint", "prop_emigrants_timepoint", "prob_emigration"]
                    )) &
                    (df["province"] == province) &
                    (df["sex"] == sex)
                ],
                y="value",
                ylabel="Proportion",
                color="series",
                title=f"Migration Proportions, Sex = {sex}, Province = {province}",
                file_path=get_data_path(
                    f"data_generation/figures/{time_delta_tag}/migration/proportions_{province}_{sex}.png",
                    mkdirs=True
                ),
                height=5000,
                width=3000
            )

    # Save the migration data to a CSV file
    file_path = get_data_path(
        f"processed_data/{time_delta_tag}/migration/migration_table.csv",
        mkdirs=True
    )
    logger.info(f"Saving data to {file_path}")
    df_migration.drop(
        columns=[
            "n_immigrants", "n_emigrants"
        ],
        inplace=True
    )
    df_migration.to_csv(file_path, index=False, date_format=DATE_FORMAT)


def plot(
    df: pd.DataFrame,
    y: str,
    color: str,
    title: str = "",
    ylabel: str = "",
    file_path: pathlib.Path | None = None,
    width: int = 2000,
    height: int = 1500
):
    """Plot the migration data for validation.
    
    Args:
        df: A dataframe containing the life table data. Must have columns:

            * ``timepoint (dt.datetime)``: The given timepoint.
            * ``province (str)``: The 2-letter province ID, e.g. ``BC``.
            * ``age (int)``: The integer age.
            * ``sex (str)``: One of ``M`` = male, ``F`` = female.
            * ``prob_death (float)``: The probability of death for the given timepoint, province,
              age, and sex.

        y: The name of the column in the dataframe which will be plotted as the ``y`` data.
        color: The name of the column in the dataframe which will be used to color the data.
        title: The title of the plot.
        ylabel: The label for the y-axis.
        file_path: The path to save the plot to.
        width: The width of the plot.
        height: The height of the plot.
    """

    fig = px.line(
        df.loc[df["province"].isin(["BC", "CA"])].dropna(),
        x="age",
        y=y,
        render_mode="svg",
        color=color,
        markers=True,
        facet_row="projection_scenario",
        facet_col="timepoint",
        facet_row_spacing=0.01,  # Shrink vertical gap to 1%
        facet_col_spacing=0.01,   # Shrink horizontal gap to 1%
        title=title,
        labels={y: ylabel}
    )
    fig.update_yaxes(matches=None)
    
    fig.update_layout(
        font=dict(size=30),
        title=dict(font=dict(size=50)),
        showlegend=True,
        width=width,
        height=height,
        autosize=False,
        margin=dict(l=120, r=220, t=120, b=120)
    )

    fig.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.split("=")[-1],
            font=dict(size=30),
            textangle=0,
        )
    )

    fig.write_image(str(file_path), scale=2, width=width, height=height)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    time_delta = TimeDelta(iso_string=args.time_delta)
    generate_migration_data(time_delta=time_delta)
