import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import pathlib
from leap.utils import get_data_path, get_time_delta_tag, TimeDelta
from leap.logger import get_logger
from leap.data_generation.utils import get_parser, split_ages
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_TIMEPOINT = dt.datetime(2000, 1, 1)
MIN_TIMEPOINT_PROJ = dt.datetime(2021, 1, 1)
MAX_TIMEPOINT = dt.datetime(2065, 1, 1)
PROVINCES = ["CA", "BC"]


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
    logger.info("Loading initial population data from CSV file...")
    time_delta_tag = get_time_delta_tag(time_delta)
    df_population = pd.read_csv(
        get_data_path(f"processed_data/{time_delta_tag}/birth/initial_population.csv"),
        parse_dates=["timepoint"]
    )
    logger.info("Loading mortality data from CSV file...")
    life_table = pd.read_csv(
        get_data_path(f"processed_data/{time_delta_tag}/life_table.csv"),
        parse_dates=["timepoint"]
    )
    if time_delta < TimeDelta(years=1):
        life_table = split_ages(life_table, time_delta, TimeDelta(years=1), [])
        df_population = split_ages(df_population, time_delta, TimeDelta(years=1), ["n_age", "n_birth"])

    df_migration = pd.DataFrame({
        "timepoint": np.array([], dtype=dt.datetime),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "projection_scenario": [],
        "delta_n": [],
        "prop_migrants_birth": [],
        "prop_immigrants_timepoint": [],
        "prop_emigrants_timepoint": [],
        "prob_emigration": []
    })

    for province in PROVINCES:
        logger.info(f"Processing migration data for province {province}...")

        # Select only the data for the given province and the years after the starting year
        df = df_population.loc[
            (df_population["timepoint"] >= MIN_TIMEPOINT) &
            (df_population["province"] == province)
        ]
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

        # Get the list of projection scenarios, excluding "past"
        projection_scenarios = df.loc[df["projection_scenario"] != "past", "projection_scenario"].unique()
        min_timepoint = df["timepoint"].min()
        min_age = 0

        for projection_scenario in projection_scenarios:
            logger.info(f"Projection scenario: {projection_scenario}")

            # select only the current projection scenario and the past projection scenario
            df_proj = df.loc[
                (df["projection_scenario"].isin(["past", projection_scenario])) &
                ~((df["projection_scenario"] == "past") & (df["timepoint"] == MIN_TIMEPOINT_PROJ))
            ]

            # join to the life table to get death probabilities
            df_proj = df_proj.merge(
                life_table, on=["timepoint", "age", "province", "sex"], how="left"
            )

            # get the number of births in each year
            df_birth = df_proj.loc[df_proj["age"] == 0]
            grouped_df = df_birth.groupby("timepoint")
            df_birth["n_birth"] = grouped_df.transform("sum")["n"]
            df_birth = df_birth.loc[df_birth["sex"] == "F", ["timepoint", "n_birth"]]

            # get the previous timepoint's cohort for each entry
            df_proj["age_key"] = df_proj["age"] - time_delta.total_years()
            df_proj["timepoint_key"] = df_proj["timepoint"].apply(
                lambda x: x - time_delta
            )

            df_prev = df_proj[["sex", "age", "timepoint", "n", "prob_death"]].rename(columns={
                "age": "age_key",
                "timepoint": "timepoint_key",
                "n": "n_prev",
                "prob_death": "prob_death_prev"
            })

            df_proj = df_proj.merge(df_prev, on=["sex", "age_key", "timepoint_key"], how="left")
            df_proj["age_prev"] = df_proj["age_key"]
            df_proj["timepoint_prev"] = df_proj["timepoint_key"]
            df_proj = df_proj.drop(columns=["age_key", "timepoint_key"])

            # remove the missing data
            df_proj = df_proj.dropna(subset=["n_prev"])

            # compute the signed population change due to net migration
            df_proj["delta_n"] = df_proj.apply(
                lambda x: get_delta_n(x["n"], x["n_prev"], x["prob_death_prev"]), axis=1
            )

            # number of migrants
            df_proj["n_immigrants"] = df_proj["delta_n"].clip(lower=0)
            df_proj["n_emigrants"] = (-df_proj["delta_n"]).clip(lower=0)

            # add the n_birth column to df_proj
            df_proj = pd.merge(df_proj, df_birth, on="timepoint", how="left")

            # signed proportion relative to births
            df_proj["prop_migrants_birth"] = df_proj["delta_n"] / df_proj["n_birth"]

            # timepoint proportions with separate denominators for immigration and emigration
            df_proj["n_immigrants_timepoint"] = df_proj.groupby("timepoint")["n_immigrants"].transform("sum")
            df_proj["n_emigrants_timepoint"] = df_proj.groupby("timepoint")["n_emigrants"].transform("sum")
            df_proj["prop_immigrants_timepoint"] = df_proj["n_immigrants"] / df_proj["n_immigrants_timepoint"]
            df_proj["prop_emigrants_timepoint"] = df_proj["n_emigrants"] / df_proj["n_emigrants_timepoint"]
            df_proj = df_proj.fillna(0)

            # per-person probability of emigrating
            df_proj["prob_emigration"] = df_proj["n_emigrants"] / df_proj["n"]

            df_migration_proj = df_proj.copy()

            # convert the "past" projection scenario to the given projection scenario
            df_migration_proj["projection_scenario"] = projection_scenario

            df_migration = pd.concat([df_migration, df_migration_proj], axis=0)

    return df_migration


def generate_migration_data(time_delta: TimeDelta):
    df_migration = load_migration_data(time_delta)
    time_delta_tag = get_time_delta_tag(time_delta)

    # Create validation plots for the migration data
    timepoints = df_migration["timepoint"].unique()
    indices = np.arange(start=0, stop=len(timepoints), step=max(1, len(timepoints) // 5))
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
    plot(
        df=df.loc[df["series"].isin(["n_immigrants", "n_emigrants", "delta_n"])],
        y="value",
        color="series",
        title=f"Net Migration",
        file_path=get_data_path(
            f"data_generation/figures/{time_delta_tag}/migration_delta_n.png", mkdirs=True
        ),
        height=3000
    )
    plot(
        df=df.loc[df["series"].isin(
            ["prop_immigrants_timepoint", "prop_emigrants_timepoint", "prob_emigration"]
        )],
        y="value",
        color="series",
        title=f"Migration Proportions",
        file_path=get_data_path(
            f"data_generation/figures/{time_delta_tag}/migration_proportions.png", mkdirs=True
        ),
        height=3000
    )

    # Save the migration data to a CSV file
    file_path = get_data_path(f"processed_data/{time_delta_tag}/migration_table.csv", mkdirs=True)
    logger.info(f"Saving data to {file_path}")
    df_migration.drop(
        columns=[
            "n_immigrants", "n_emigrants", "n", "n_birth", "n_immigrants_timepoint",
            "n_emigrants_timepoint"
        ],
        inplace=True
    )
    df_migration.to_csv(file_path, index=False)


def plot(
    df: pd.DataFrame,
    y: str,
    color: str,
    title: str = "",
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
        facet_col="sex",
        facet_row="timepoint",
        facet_row_spacing=0.01,  # Shrink vertical gap to 1%
        facet_col_spacing=0.01,   # Shrink horizontal gap to 1%
        title=title
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
