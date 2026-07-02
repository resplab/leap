import pathlib
import itertools
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
from typing import Tuple, Dict
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


MIN_TIMEPOINT = dt.datetime(1996, 1, 1)
MAX_TIMEPOINT = dt.datetime(2068, 12, 31)

TIME_DELTA_OD = TimeDelta(years=1) # original time delta of the data


def convert_prob_death(
    prob_death: float,
    time_delta_od: TimeDelta,
    time_delta: TimeDelta
) -> float:
    r"""Convert the probability of death from the original time delta to the new time delta.

    The probability of death, :math:`q(x, \Delta x)`, depends on the choice of :math:`\Delta x`,
    i.e. ``time_delta``. The original data is collected with :math:`\Delta x = 1` year, but we may
    want to convert it to a different :math:`\Delta x`, e.g. 1 month.

    Assuming the hazard ratio is constant over the time interval:

    .. math::

        q(x, \Delta x_b, t) = 1 - (1 - q(x, \Delta x_a, t))^{\frac{\Delta x_b}{\Delta x_a}}

    Args:
        prob_death: The probability of death for the original time delta, :math:`q(x, \Delta x_a)`.
        time_delta_od: The original time delta, :math:`\Delta x_a`.
        time_delta: The new time delta, :math:`\Delta x_b`.

    Returns:
        The probability of death for the new time delta, :math:`q(x, \Delta x_b)`.
    """

    if time_delta_od == time_delta:
        return prob_death

    return 1 - (1 - prob_death)**(time_delta.total_years() / time_delta_od.total_years())


def calculate_life_expectancy(life_table: pd.DataFrame, time_delta: TimeDelta) -> float:
    """Determine the life expectancy for a person born in a given time interval.

    The life expectancy can be calculated from the death probability using the formulae
    delineated here:
    `Life Table Definitions <https://www.ssa.gov/oact/HistEst/CohLifeTables/LifeTableDefinitions.pdf>`_
    
    Args:
        life_table: A dataframe containing the probability of death for a single timepoint,
            province and sex, for each age. Columns:

            * ``age``: the integer age.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``timepoint``: the timepoint of the data in the row.
            * ``province``: A string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``prob_death``: the probability of death for a given age, province, sex, and timepoint.
        time_delta: The duration of time between data points.

    Returns:
        The life expectancy for a person born in the given time interval, in a given province,
        for a given sex.
    """
    assert life_table["sex"].nunique() == 1, "Dataframe should only contain one sex."
    assert life_table["province"].nunique() == 1, "Dataframe should only contain one province."

    df = life_table.sort_values("age").copy()
    df.set_index("age", inplace=True)

    # l(x): calculate the number of people alive up to age x
    n_alive_by_age_0 = 100000 # l(0): initial number of people at age 0
    n_alive_by_age = [] # l(x)
    n_alive_by_age_prev = n_alive_by_age_0
    prob_death_prev = 0
    for age in df.index:
        if age == 0:
            n_alive_by_age.append(n_alive_by_age_0)
        else:
            # l(x) = l(x-dx) * (1 - q(x-dx)); q(x) = prob_death at age [x, x+dx)
            n_alive_by_age.append(
                n_alive_by_age_prev * (1 - prob_death_prev)
            )
        n_alive_by_age_prev = n_alive_by_age[-1]
        prob_death_prev = df.loc[age, "prob_death"]

    df["n_alive_by_age"] = n_alive_by_age

    # L(x): calculate the number of person-years lived between ages [x, x+dx)
    # L(x) = (l(x) - a(x) * d(x)) * dx
    # d(x) = l(x) * q(x)
    df["a_x"] = [0.5] * df.shape[0]
    if time_delta > TimeDelta(days=7):
        df.loc[0, "a_x"] = 0.1
    df["n_person_years_interval"] = df.apply(
        lambda x: (x["n_alive_by_age"] - x["a_x"] * x["prob_death"] * x["n_alive_by_age"]) * time_delta.total_years(),
        axis=1
    )

    # L(x_f): calculate the number of person-years lived between ages [x_f, infinity)
    max_age = df.index.max()
    factor = 1.4 * TIME_DELTA_OD.total_years() / time_delta.total_years()
    df.loc[max_age, "n_person_years_interval"] = df.loc[max_age, "n_alive_by_age"] * factor

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
    r"""Given the (known) prob death for a past timepoint, calculate the prob death at a future timepoint.

    .. math::

        \sigma^{-1}(p(\text{sex}, \text{age}, \text{timepoint})) =
            \sigma^{-1}(p(\text{sex}, \text{age}, \text{timepoint}_0)) - 
            \beta(\text{sex})(\text{timepoint} - \text{timepoint}_0)

    Args:
        prob_death: The probability of death for ``timepoint_initial``, the last timepoint that past
            data was collected, for a given age, sex, province, and projection scenario.
        timepoint_initial: The initial timepoint with a known probability of death. This is the last
            timepoint that the past data was collected.
        timepoint: The current timepoint.
        beta_time: The beta parameter for the given sex, province, and projection scenario.

    Returns:
        The projected probability of death for the current timepoint.

    Examples:

        >>> timepoint_initial = dt.datetime(1996, 1, 1)
        >>> timepoint = dt.datetime(2026, 1, 1)
        >>> get_prob_death_projected(0.01, timepoint_initial, timepoint, -0.02)
        np.float64(0.005512990331820702)
    """
    time_diff = TimeDelta(dt1=timepoint, dt2=timepoint_initial).total_years()
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
    """Get the life table for a single timepoint.

    Args:
        beta_time: The beta parameter for the given timepoint.
        life_table: A dataframe containing the projected probability of death
            for the starting timepoint, for a given sex and province. Columns:

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
    timepoint_initial: dt.datetime,
    projection_scenario: str,
    time_delta: TimeDelta
) -> np.ndarray:
    """Calculate the difference between the projected life expectancy and desired life expectancy.

    This function is passed to the ``scipy.optimize.leastsq`` function. We want to find ``beta_time``
    such that the projected life expectancy is as close as possible to the desired life expectancy.
    
    Args:
        beta_time: The beta parameter for the given timepoint. The ``scipy.optimize.leastsq``
            function requires that this be a 1D array, but we only have a single parameter.
        life_table: A dataframe containing the projected probability of death
            for the calibration year, for a given sex and province. Columns:

            * ``age``: the integer age.
            * ``sex``: one of ``M`` = male, ``F`` = female.
            * ``timepoint``: the calibration timepoint.
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


def load_past_death_data() -> pd.DataFrame:
    """Load the past death data from the ``StatCan`` CSV file.
    
    Returns:
        A dataframe containing the probability of death and the standard error for each timepoint,
        province, age, and sex. The time delta of the data is that of the original data, which is
        1 year.
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
        df["timepoint"] >= MIN_TIMEPOINT,
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

    df.sort_values(["province", "age", "sex", "timepoint"], inplace=True)
    df = df[["province", "age", "sex", "timepoint", "prob_death", "se"]]

    df["projection_scenario"] = ["past"] * df.shape[0]

    return df


def load_projected_death_data(min_timepoint: dt.datetime) -> pd.DataFrame:
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

    df = df.loc[df["timepoint"] >= min_timepoint].reset_index(drop=True)

    return df


def compute_beta_parameters(
    past_life_table: pd.DataFrame,
    df_calibration: pd.DataFrame,
    time_delta_od: TimeDelta = TIME_DELTA_OD,
    x0: float = -0.02,
    xtol: float = 0.00001   
) -> Dict[Tuple[str, str, str], float]:
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

        time_delta_od: The original duration of time between data points in the past data.
        x0: The initial guess for the beta parameter.
        xtol: The tolerance for the beta parameter.
    
    Returns:
        A dictionary containing the beta parameters for each province, sex, and projection scenario.
    """
    keys = list(itertools.product(
        past_life_table["province"].unique(),
        past_life_table["sex"].unique(), 
        past_life_table["projection_scenario"].unique())
    )
    beta_parameters = {key: 0.0 for key in keys}

    for key in keys:
        province = key[0]
        sex = key[1]
        projection_scenario = key[2]

        life_table = past_life_table[past_life_table["province"] == province]
        max_timepoint_past = life_table["timepoint"].max()
        life_table = life_table[life_table["timepoint"] == max_timepoint_past]

        beta_time = optimize.leastsq(
            compute_life_expectancy_diff,
            x0=[x0],
            args=(
                life_table,
                df_calibration,
                sex,
                province,
                max_timepoint_past,
                projection_scenario,
                time_delta_od
            ),
            xtol=xtol,
        )[0][0]

        beta_parameters[key] = beta_time

    return beta_parameters


def get_projected_death_data(
    beta_parameters: Dict[Tuple[str, str, str], float],
    past_life_table: pd.DataFrame
) -> pd.DataFrame:
    """Load the projected death data from ``StatCan`` CSV file.
    
    Args:
        beta_parameters: A dictionary containing the beta parameters for each province, sex, and
            projection scenario.
        past_life_table: A dataframe containing the probability of death and the standard error
            for each timepoint, province, age, and sex. Columns:
            
            * ``timepoint``: the starting timepoint of the interval during which the data was collected.
            * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``projection_scenario``: The projection scenario, i.e. ``"past"``.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``age``: the integer age.
            * ``prob_death``: the probability of death.
            * ``se``: the standard error of the probability of death.
    
    Returns:
        A dataframe containing the predicted probability of death and the standard error
        for each annual timepoint, province, age, and sex.
        Columns:

        * ``timepoint``: The starting timepoint of the interval the data applies to.
        * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``projection_scenario``: The projection scenario, e.g. ``"M3"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``prob_death``: The probability that a person of the given age, sex, and province
          will die in the given year.
        * ``se``: The standard error of the probability of death.
    """

    max_timepoint_past = past_life_table["timepoint"].max()
    starting_timepoint = max_timepoint_past + TIME_DELTA_OD

    projected_life_table = pd.DataFrame(
        data=list(itertools.product(
            set([key[0] for key in beta_parameters.keys()]),
            set([key[1] for key in beta_parameters.keys()]),
            set([key[2] for key in beta_parameters.keys()]),
            list(date_range(starting_timepoint, MAX_TIMEPOINT, TIME_DELTA_OD)),
            past_life_table["age"].unique(),
            [0.0]
        )),
        columns=["province", "sex", "projection_scenario", "timepoint", "age", "prob_death"]
    )

    projected_life_table = pd.merge(
        projected_life_table,
        past_life_table.loc[past_life_table["timepoint"] == max_timepoint_past],
        how="left",
        on=["province", "sex", "age", "projection_scenario"],
        suffixes=("", "_initial")
    )

    projected_life_table["prob_death"] = projected_life_table.apply(
        lambda x: get_prob_death_projected(
            prob_death=x["prob_death_initial"],
            timepoint_initial=starting_timepoint,
            timepoint=x["timepoint"],
            beta_time=beta_parameters[(x["province"], x["sex"], x["projection_scenario"])]
        ),
        axis=1
    )

    projected_life_table.sort_values(["province", "projection_scenario", "age", "sex", "timepoint"], inplace=True)
    projected_life_table = projected_life_table[
        ["province", "projection_scenario", "age", "sex", "timepoint", "prob_death", "se"]
    ]

    return projected_life_table



def generate_death_data(
    time_delta: TimeDelta,
    to_csv: bool = True,
    draw_plot: bool = False,
    x0: float = -0.02,
    xtol: float = 0.00001
) -> None | pd.DataFrame:
    """Generate the mortality data CSV.
    
    Args:
        time_delta: The duration of time between data points.
        to_csv: Whether to save the data to a CSV file. If False, the dataframe will be returned
            instead.
        draw_plot: Whether to draw a plot of the mortality data for validation.
        x0: The initial guess for the beta parameter.
        xtol: The tolerance for the beta parameter.

    Returns:
        If ``to_csv`` is False, a dataframe containing the probability of death and the standard
        error for each timepoint, province, age, and sex.
    """
    past_life_table = load_past_death_data()
    df_calibration = load_projected_death_data(min_timepoint=past_life_table["timepoint"].max() + time_delta)

    # Compute the beta parameters for each province, sex, and projection scenario
    beta_parameters = compute_beta_parameters(
        past_life_table=past_life_table,
        df_calibration=df_calibration,
        time_delta_od=TIME_DELTA_OD,
        x0=x0,
        xtol=xtol
    )

    # Get the projected death data for each province, sex, projection scenario, and year
    projected_life_table = get_projected_death_data(
        beta_parameters, past_life_table
    )
    life_table = pd.concat([past_life_table, projected_life_table], axis=0)
    life_table["age_int"] = life_table["age"].apply(lambda x: int(x))
    grouped_df = life_table.groupby(["age_int", "province", "sex", "timepoint"], as_index=False).agg({
        "prob_death": "mean",
        "se": lambda x: np.sqrt(np.sum(x**2)) / len(x)
    })
    life_table = grouped_df[["province", "age_int", "sex", "timepoint", "prob_death", "se"]]
    life_table.rename(columns={"age_int": "age"}, inplace=True)
    life_table.sort_values(["province", "sex", "timepoint", "age"], inplace=True)

    time_delta_tag = get_time_delta_tag(time_delta)

    if draw_plot:
        plot(
            life_table.loc[life_table["age"].isin([0, 10, 20, 40, 60, 80, 100])].copy(),
            y="prob_death",
            color="age",
            title="Mortality Data",
            file_path=get_data_path(
                f"data_generation/figures/{time_delta_tag}/life_expectancy.png",
                mkdirs=True
            )
        )

    # save the data
    if to_csv:
        file_path = get_data_path(f"processed_data/{time_delta_tag}/life_table.csv", mkdirs=True)
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
        df.loc[df["province"].isin(["BC", "CA"])],
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