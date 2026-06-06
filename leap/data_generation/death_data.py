import pathlib
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
from scipy import optimize
from leap.utils import get_data_path
from leap.logger import get_logger
from leap.data_generation.utils import format_age_group, get_province_id, get_sex_id, get_parser, \
    interpolate
from leap.utils import TimeDelta, date_range, get_time_delta_tag
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


STARTING_TIMEPOINT = dt.datetime(1996, 1, 1)
FINAL_TIMEPOINT = dt.datetime(2068, 12, 31)

TIME_DELTA_OD = TimeDelta(years=1) # original time delta of the data



def calculate_life_expectancy(life_table: pd.DataFrame, time_delta: TimeDelta) -> float:
    """Determine the life expectancy for a person born in a given year.

    The life expectancy can be calculated from the death probability using the formulae
    delineated here:
    `Life Table Definitions <https://www.ssa.gov/oact/HistEst/CohLifeTables/LifeTableDefinitions.pdf>`_
    
    Args:
        life_table: A dataframe containing the probability of death for a single year,
            province and sex, for each age. Columns:

            * ``age``: the integer age.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``year``: the integer calendar year.
            * ``province``: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
            * ``prob_death``: the probability of death for a given age, province, sex, and year.
        time_delta: The duration of time between data points.

    Returns:
        The life expectancy for a person born in the given year, in a given province,
        for a given sex.
    """

    df = life_table.sort_values("age").copy()
    df.set_index("age", inplace=True)

    # l(x): calculate the number of people alive up to age x
    n_alive_by_age_0 = 100000 # l(0): initial number of people at age 0
    n_alive_by_age = [] # l(x)
    for age in df.index:
        if age == 0:
            n_alive_by_age.append(n_alive_by_age_0)
        else:
            # l(x) = l(x-1) * (1 - q(x)-1); q(x) = prob_death at age x
            n_alive_by_age.append(
                n_alive_by_age[age - 1] * (1 - df.loc[age - 1, "prob_death"])
            )
    df["n_alive_by_age"] = n_alive_by_age

    # L(x): calculate the number of person-years lived between ages [x, x+dx)
    # L(x) = (l(x) - 0.5 * d(x)) * dx
    # d(x) = l(x) * q(x)
    df["n_person_years_interval"] = df.apply(
        lambda x: (x["n_alive_by_age"] - 0.5 * x["prob_death"] * x["n_alive_by_age"]) * time_delta.total_years(), axis=1
    )

    # L(0): calculate the number of person-years lived between ages [0, 1)
    # L(0) = L(1) - f(0) * d(0)
    # d(0) = l(0) * q(0)
    df.loc[0, "n_person_years_interval"] = (
        df.loc[1, "n_person_years_interval"] +
        0.1 * df.loc[0, "prob_death"] * n_alive_by_age_0
    )

    # L(110): calculate the number of person-years lived between ages [110, 111)
    df.loc[110, "n_person_years_interval"] = df.loc[110, "n_alive_by_age"] * 1.4

    # T(x): calculate the total number of person-years lived after age x
    # T(x) = sum(L(x) for x in [x, 110])
    df["n_person_years_after_age"] = df["n_person_years_interval"].sort_index(
        ascending=False
    ).cumsum().sort_index()

    # E(x): calculate the number of years left to live at age x
    # E(x) = T(x) / l(x)
    df["n_years_left_at_age"] = df.apply(
        lambda x: x["n_person_years_after_age"] / x["n_alive_by_age"], axis=1
    )

    # E(0): calculate the number of years left to live at age 0, aka life expectancy
    life_expectancy = df.loc[0, "n_years_left_at_age"]

    return life_expectancy


def get_prob_death_projected(
    prob_death: float,
    timepoint_initial: dt.datetime,
    timepoint: dt.datetime,
    beta_time: float
) -> float:
    r"""Given the (known) prob death for a past year, calculate the prob death in a future year.

    .. math::

        \sigma^{-1}(p(\text{sex}, \text{age}, \text{year})) =
            \sigma^{-1}(p(\text{sex}, \text{age}, \text{year}_0)) - 
            \beta(\text{sex})(\text{year} - \text{year}_0)

    Args:
        prob_death: The probability of death for ``year_initial``, the last year that past data was
            collected, for a given age, sex, province, and projection scenario.
        timepoint_initial: The initial timepoint with a known probability of death. This is the last
            timepoint that the past data was collected.
        timepoint: The current timepoint.
        beta_time: The beta parameter for the given sex, province, and projection scenario.

    Returns:
        The projected probability of death for the current year.
    """
    time_diff = TimeDelta(dt1=timepoint_initial, dt2=timepoint).total_years()
    prob_death = min(prob_death, 0.9999999999)
    odds = (prob_death / (1 - prob_death)) * np.exp(time_diff * beta_time)
    prob_death_projected = max(min(odds / (1 + odds), 1), 0)
    return prob_death_projected



def get_projected_life_table_single_timepoint(
    beta_time: float,
    life_table: pd.DataFrame,
    timepoint_initial: dt.datetime,
    timepoint: dt.datetime,
    sex: str,
    province: str
) -> pd.DataFrame:
    """Get the life table for a single year.

    Args:
        beta_time: The beta parameter for the given timepoint.
        life_table: A dataframe containing the projected probability of death
            for the starting year, for a given sex and province. Columns:

            * ``age``: the integer age.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``timepoint``: the starting calendar year.
            * ``province``: a string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``prob_death``: the probability of death for a given age, province, sex, and year.

        timepoint_initial: The initial year with a known probability of death. This is the last year
            that the past data was collected.
        timepoint: The current timepoint.
        sex: One of ``M`` = male, ``F`` = female.
        province: a string indicating the province abbreviation, e.g. ``"BC"``.
            For all of Canada, set province to ``"CA"``.

    Returns:
        A dataframe containing the projected probability of death for the given timepoint,
        sex, and province.
    """
    df = life_table.loc[(life_table["sex"] == sex) & (life_table["province"] == province)].copy()
    df["prob_death_proj"] = df["prob_death"].apply(
        lambda x: get_prob_death_projected(x, timepoint_initial, timepoint, beta_time)
    )

    df["timepoint"] = [timepoint] * df.shape[0]

    df["se"] = df.apply(
        lambda x: (x["prob_death_proj"] * x["se"]) / x["prob_death"], axis=1
    )
    df.drop(columns=["prob_death"], inplace=True)
    df.rename(columns={"prob_death_proj": "prob_death"}, inplace=True)

    return df


def compute_life_expectancy_diff(
    beta_time: np.ndarray,
    life_table: pd.DataFrame,
    df_calibration: pd.DataFrame,
    sex: str,
    province: str, 
    timepoint_initial: int,
    time_delta: TimeDelta
) -> np.ndarray:
    """Calculate the difference between the projected life expectancy and desired life expectancy.

    This function is passed to the ``scipy.optimize.brentq`` function. We want to find ``beta_year``
    such that the projected life expectancy is as close as possible to the desired life expectancy.
    
    Args:
        beta_time: The beta parameter for the given timepoint. The ``scipy.optimize.leastsq``
            function requires that this be a 1D array, but we only have a single parameter.
        life_table: A dataframe containing the projected probability of death
            for the calibration year, for a given sex and province. Columns:

            * ``age``: the integer age.
            * ``sex``: one of ``M`` = male, ``F`` = female.
            * ``timepoint``: the calibration calendar year.
            * ``province``: a 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``prob_death``: the probability of death for a given age, province, sex, and year.

        df_calibration: A dataframe containing the life expectancy projections for the calibration
            years. Columns:

            * ``year``: The calendar year. Range ``[1988, 2073]``.
            * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``sex``: One of ``F`` = female, ``M`` = male.
            * ``projection_scenario``: The projection scenario, e.g. ``"M3"``.
            * ``mortality_scenario``: The mortality scenario. One of:
                - ``LM``: Low mortality
                - ``MM``: Medium mortality
                - ``HM``: High mortality
            * ``life_expectancy``: The life expectancy in years for the given year, province,
              sex, projection scenario, and mortality scenario.

        sex: one of ``M`` = male, ``F`` = female.
        province: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
            For all of Canada, set province to ``"CA"``.
        timepoint_initial: The initial timepoint with a known probability of death. This is the last
            timepoint that the past data was collected.
        projection_scenario: The projection scenario, e.g. ``"M3"``.
        time_delta: The duration of time between data points.
    
    Returns:
        The difference between the projected life expectancy of the calibration year
        and the desired life expectancy, for each of the calibration years.
    """
    beta_time = beta_time[0]
    desired_life_expectancies = df_calibration.loc[
        (df_calibration["sex"] == sex) &
        (df_calibration["province"] == province) &
        (df_calibration["projection_scenario"] == projection_scenario)
    ]

    diff = []
    for timepoint in desired_life_expectancies["timepoint"]:
        projected_life_table = get_projected_life_table_single_timepoint(
            beta_time, life_table, timepoint_initial, timepoint, sex, province
        )
        logger.info(f"Calculating life expectancy for {timepoint}, {sex}, {province}, beta_time={beta_time}")

        life_expectancy = calculate_life_expectancy(projected_life_table, time_delta)
        desired_life_expectancy = desired_life_expectancies.loc[
            desired_life_expectancies["timepoint"] == timepoint, "life_expectancy"
        ].values[0]
        diff.append(np.abs(life_expectancy - desired_life_expectancy))
    
    return np.array(diff)


def load_past_death_data(time_delta: TimeDelta) -> pd.DataFrame:
    """Load the past death data from the ``StatCan`` CSV file.

    Args:
        time_delta: The duration of time between data points.
    
    Returns:
        A dataframe containing the probability of death and the standard error
        for each timepoint, province, age, and sex.
        Columns:

        * ``timepoint``: The starting timepoint of the interval during which the data was collected.
        * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``prob_death``: The probability that a person of the given age, sex, and province
          will die in the given year.
        * ``se``: The standard error of the probability of death.
    """ 
    logger.info("Loading mortality data from CSV file...")
    df = pd.read_csv(get_data_path("original_data/13100837.csv"), parse_dates=["REF_DATE"])

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename the columns
    df.rename(
        columns={"REF_DATE": "timepoint", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age"},
        inplace=True
    )

    # select the required columns
    df = df.loc[
        df["timepoint"] >= STARTING_TIMEPOINT,
        ["timepoint", "province", "sex", "age", "ELEMENT", "VALUE"]
    ]

    # format the age group into an integer age
    df["age"] = df["age"].apply(lambda x: format_age_group(x, "110 years and over"))

    # convert province names to 2-letter province IDs
    df["province"] = df["province"].apply(get_province_id)

    # filter only "CA" and "BC"
    df = df.loc[df["province"].isin(["CA", "BC"])]

    # convert sex to 1-letter ID ("F", "M", "B")
    df["sex"] = df["sex"].apply(get_sex_id)

    # remove sex category "Both"
    df = df.loc[df["sex"] != "B"]

    # select only the "qx" elements, which relate to the probability of death and the SE
    df = df.loc[df["ELEMENT"].str.contains("qx")]

    # create a df with the probability of death
    df_prob = df.loc[df["ELEMENT"].str.contains("Death probability between age")]
    df_prob = df_prob.drop(columns=["ELEMENT"])
    df_prob.rename(columns={"VALUE": "prob_death"}, inplace=True)

    # create a df with the standard error of the probability of death
    df_se = df.loc[df["ELEMENT"].str.contains("Margin of error")]
    df_se = df_se.drop(columns=["ELEMENT"])
    df_se.rename(columns={"VALUE": "se"}, inplace=True)

    # join the two tables
    df = pd.merge(df_prob, df_se, on=["timepoint", "province", "sex", "age"], how="left")

    # Interpolate the birth estimates for the missing timepoints in the past data
    df = interpolate(
        data=df.copy().reset_index(drop=True),
        col_pred="prob_death",
        time_delta=time_delta,
        time_delta_od=TIME_DELTA_OD,
        columns_group=["province", "age", "sex"]
    ).reset_index(drop=True)
    df.sort_values(["province", "age", "sex", "timepoint"], inplace=True)
    df = df[["province", "age", "sex", "timepoint", "prob_death", "se"]]

    if time_delta < TimeDelta(years=1):
        n_intervals = TimeDelta(years=1) // time_delta
        df = df.loc[df.index.repeat(n_intervals)].reset_index(drop=True)
        df["age"] = df["age"] + np.arange(len(df)) % n_intervals / n_intervals

    return df


def load_projected_death_data() -> pd.DataFrame:
    """Load the projected death data from the ``StatCan`` CSV files.

    ``Statistics Canada`` provides two tables with life expectancy projections:

    - `Table 3.2 (Canada) <https://www150.statcan.gc.ca/n1/pub/91-620-x/91-620-x2025001-eng.html>`_
    - `Table 5.2 (Provinces) <https://www150.statcan.gc.ca/n1/pub/91-620-x/91-620-x2025002-eng.html>`_

    This data is only available for selected years.
    
    Returns:
        A dataframe containing the life expectancy from selected calibration years from
        ``Statistics Canada``:

        * ``timepoint (dt.datetime)``: Range ``[1988, 2073]``.
        * ``province (str)``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex (str)``: One of ``F`` = female, ``M`` = male.
        * ``projection_scenario (str)``: The projection scenario, e.g. ``"M3"``.
        * ``mortality_scenario (str)``: The mortality scenario. One of:
            - ``LM``: Low mortality
            - ``MM``: Medium mortality
            - ``HM``: High mortality
        * ``life_expectancy (float)``: The life expectancy in years for the given year, province,
          sex, projection scenario, and mortality scenario.
    """

    # Load the life expectancy projections for Canada from StatCan
    df_can = pd.read_csv(get_data_path("original_data/mortality_projections_table_3-2.csv"))
    df_can = df_can.melt(
        id_vars=["year", "sex"],
        value_vars=["LG", "M1", "M2", "M3", "M4", "M5", "M6", "HG", "SA", "FA"],
        var_name="projection_scenario",
        value_name="life_expectancy"
    )
    df_can["year"] = df_can["year"].apply(lambda x: int(x.split("/")[0]))
    df_can["province"] = ["CA"] * df_can.shape[0]

    # Load the life expectancy projections for the provinces / territories from StatCan
    df_prov = pd.read_csv(get_data_path("original_data/mortality_projections_table_5-2.csv"))
    df_prov = df_prov.melt(
        id_vars=["province", "sex", "mortality_scenario"],
        value_vars=[x for x in df_prov.columns if x.startswith("19") or x.startswith("20")],
        var_name="year",
        value_name="life_expectancy"
    )
    df_prov["year"] = df_prov["year"].apply(lambda x: int(x.split("/")[0]))

    # Load the projection / mortality scenario mappings
    df_scenarios = pd.read_csv(get_data_path("original_data/mortality_projections_table_3-1.csv"))
    df_can = pd.merge(
        df_can,
        df_scenarios[["projection_scenario", "mortality_scenario"]],
        on="projection_scenario",
        how="left"
    )
    df_prov = pd.merge(
        df_prov,
        df_scenarios[["projection_scenario", "mortality_scenario"]],
        on="mortality_scenario",
        how="left"
    )

    # Combine the dataframes
    df = pd.concat([df_can, df_prov], axis=0)

    # Remove NA columns
    df = df.dropna(subset=["life_expectancy"])

    # Convert year to timepoint
    df["timepoint"] = df["year"].apply(lambda x: dt.datetime(x, 1, 1))
    df = df.drop(columns=["year"])

    return df


def get_projected_death_data(
    past_life_table: pd.DataFrame,
    df_calibration: pd.DataFrame,
    time_delta: TimeDelta,
    projection_scenario: str = "M3",
    x0: float = -0.02,
    xtol: float = 0.00001
) -> pd.DataFrame:
    """Load the projected death data from ``StatCan`` CSV file.
    
    Args:
        past_life_table: A dataframe containing the probability of death and the standard error
            for each timepoint, province, age, and sex. Columns:
            
            * ``timepoint``: the starting timepoint of the interval during which the data was collected.
            * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``age``: the integer age.
            * ``prob_death``: the probability of death.
            * ``se``: the standard error of the probability of death.

        df_calibration: A dataframe containing the life expectancy projections for the calibration
            years. Columns:

            * ``timepoint``: The calendar year. Range ``[1988, 2073]``.
            * ``province``: A 2-letter string indicating the province abbreviation, e.g.
              ``"BC"``. For all of Canada, set province to ``"CA"``.
            * ``sex``: One of ``F`` = female, ``M`` = male.
            * ``projection_scenario``: The projection scenario, e.g. ``"M3"``.
            * ``mortality_scenario``: The mortality scenario. One of:
                - ``LM``: Low mortality
                - ``MM``: Medium mortality
                - ``HM``: High mortality
            * ``life_expectancy``: The life expectancy in years for the given year, province,
              sex, projection scenario, and mortality scenario.

        time_delta: The duration of time between data points.
        x0: The initial guess for the beta parameter.
        xtol: The tolerance for the beta parameter.
    
    Returns:
        A dataframe containing the predicted probability of death and the standard error
        for each timepoint, province, age, and sex.
        Columns:

        * ``timepoint``: The starting timepoint of the interval the data applies to.
        * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``prob_death``: The probability that a person of the given age, sex, and province
          will die in the given year.
        * ``se``: The standard error of the probability of death.
    """

    projected_life_table = pd.DataFrame({
        "timepoint": np.array([], dtype=dt.datetime),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "prob_death": [],
        "se": []
    })
    for province in past_life_table["province"].unique():
        life_table = past_life_table[past_life_table["province"] == province]
        max_timepoint_past = life_table["timepoint"].max()
        starting_timepoint = max_timepoint_past + time_delta
        life_table = life_table[life_table["timepoint"] == max_timepoint_past]

        beta_time_female = optimize.leastsq(
            compute_life_expectancy_diff,
            x0=[x0],
            args=(
                life_table,
                df_calibration,
                "F",
                province,
                max_timepoint_past,
                projection_scenario,
                time_delta
            ),
            xtol=xtol,
        )[0][0]

        beta_time_male = optimize.leastsq(
            compute_life_expectancy_diff,
            x0=[x0],
            args=(
                life_table,
                df_calibration,
                "M",
                province,
                max_timepoint_past,
                projection_scenario,
                time_delta
            ),
            xtol=xtol
        )[0][0]

        projected_life_table_province = pd.DataFrame({
            "timepoint": np.array([], dtype=dt.datetime),
            "province": [],
            "age": np.array([], dtype=int),
            "sex": [],
            "prob_death": [],
            "se": []
        })
        for timepoint in date_range(starting_timepoint, FINAL_TIMEPOINT + time_delta, time_delta):
            # get the prob_death projections for the year and add to dataframe
            df_female = get_projected_life_table_single_timepoint(
                beta_time_female, life_table, starting_timepoint - time_delta, timepoint, "F", province
            )
            df_male = get_projected_life_table_single_timepoint(
                beta_time_male, life_table, starting_timepoint - time_delta, timepoint, "M", province
            )
            # combine the dataframes
            projected_life_table_single_timepoint = pd.concat([df_female, df_male], axis=0)
            projected_life_table_province = pd.concat(
                [projected_life_table_province, projected_life_table_single_timepoint],
                axis=0
            )

        projected_life_table = pd.concat(
            [projected_life_table, projected_life_table_province],
            axis=0
        )

    projected_life_table.sort_values(["province", "age", "sex", "timepoint"], inplace=True)
    projected_life_table = projected_life_table[
        ["province", "age", "sex", "timepoint", "prob_death", "se"]
    ]

    return projected_life_table



def generate_death_data(
    time_delta: TimeDelta, to_csv: bool = True, draw_plot: bool = False
) -> None | pd.DataFrame:
    """Generate the mortality data CSV."""
    past_life_table = load_past_death_data(time_delta)
    df_calibration = load_projected_death_data()
    projected_life_table = get_projected_death_data(past_life_table, df_calibration, time_delta)
    life_table = pd.concat([past_life_table, projected_life_table], axis=0)

    time_delta_tag = get_time_delta_tag(time_delta)

    if draw_plot:
        plot(
            life_table.loc[life_table["age"].isin([0, 10, 20, 40, 60, 80, 100])].copy(),
            y="prob_death",
            color="age",
            title="Mortality Data",
            file_path=get_data_path(f"data_generation/figures/{time_delta_tag}/life_expectancy.png")
        )

    # save the data
    if to_csv:
        file_path = get_data_path(f"processed_data/{time_delta_tag}/life_table.csv")
        logger.info(f"Saving data to {file_path}")
        life_table.to_csv(file_path, index=False)
    else:
        return life_table
    




def plot(
    df: pd.DataFrame,
    y: str,
    color: str,
    title: str = "",
    file_path: pathlib.Path | None = None,
    width: int = 2000,
    height: int = 1500
):
    """Plot the mortality data for validation.
    
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
        x="timepoint",
        y=y,
        render_mode="svg",
        color=color,
        markers=True,
        facet_col="province",
        facet_row="sex",
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
    generate_death_data(time_delta=time_delta, to_csv=True, draw_plot=True)