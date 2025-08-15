import pandas as pd
import numpy as np
from scipy import optimize
from leap.utils import get_data_path
from leap.logger import get_logger
from leap.data_generation.utils import format_age_group, get_province_id, get_sex_id
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


STARTING_YEAR = 1996
FINAL_YEAR = 2068

# The projected life expectencies as given by StatCan for the M3 projection scenario
DESIRED_LIFE_EXPECTANCIES = pd.DataFrame({
    "province": ["CA", "CA", "BC", "BC"],
    "sex": ["M", "F", "M", "F"],
    "life_expectancy": [87.0, 90.1, 84.6, 88.0]
})

CALIBRATION_YEARS = {
    "CA": 2068,
    "BC": 2043,
    "AB": None,
    "SK": None,
    "MB": None,
    "ON": None,
    "QC": None,
    "NL": None,
    "NS": None,
    "NB": None,
    "PE": None,
    "YT": None,
    "NT": None,
    "NU": None
}


def calculate_life_expectancy(life_table: pd.DataFrame) -> float:
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

    # L(x): calculate the number of person-years lived between ages [x, x+1)
    # L(x) = l(x) - 0.5 * d(x)
    # d(x) = l(x) * q(x)
    df["n_person_years_interval"] = df.apply(
        lambda x: x["n_alive_by_age"] - 0.5 * x["prob_death"] * x["n_alive_by_age"], axis=1
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
    prob_death: float, year_initial: int, year: int, beta_year: float
) -> float:
    r"""Given the (known) prob death for a past year, calculate the prob death in a future year.

    .. math::

        \sigma^{-1}(p(\text{sex}, \text{age}, \text{year})) =
            \sigma^{-1}(p(\text{sex}, \text{age}, \text{year}_0)) - 
            e^{\beta(\text{sex})(\text{year} - \text{year}_0)}

    Args:
        prob_death: The probability of death for ``year_initial``, the last year that past data was
            collected, for a given age, sex, province, and projection scenario.
        year_initial: The initial year with a known probability of death. This is the last year
            that the past data was collected.
        year: The current year.
        beta_year: The beta parameter for the given sex, province, and projection scenario.

    Returns:
        The projected probability of death for the current year.
    """
    prob_death = min(prob_death, 0.9999999999)
    odds = (prob_death / (1 - prob_death)) * np.exp((year - year_initial) * beta_year)
    prob_death_projected = max(min(odds / (1 + odds), 1), 0)
    return prob_death_projected



def get_projected_life_table_single_year(
    beta_year: float, life_table: pd.DataFrame, year_initial: int, year: int,
    sex: str, province: str
) -> pd.DataFrame:
    """Get the life table for a single year.

    Args:
        beta_year: The beta parameter for the given year.
        life_table: A dataframe containing the projected probability of death
            for the starting year, for a given sex and province. Columns:

            * ``age``: the integer age.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``year``: the starting calendar year.
            * ``province``: a string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``prob_death``: the probability of death for a given age, province, sex, and year.

        year_initial: The initial year with a known probability of death. This is the last year
            that the past data was collected.
        year: The current year.
        sex: One of ``M`` = male, ``F`` = female.
        province: a string indicating the province abbreviation, e.g. ``"BC"``.
            For all of Canada, set province to ``"CA"``.

    Returns:
        A dataframe containing the projected probability of death for the given year,
        sex, and province.
    """
    df = life_table.loc[(life_table["sex"] == sex) & (life_table["province"] == province)].copy()
    df["prob_death_proj"] = df["prob_death"].apply(
        lambda x: get_prob_death_projected(x, year_initial, year, beta_year)
    )

    df["year"] = [year] * df.shape[0]

    df["se"] = df.apply(
        lambda x: (x["prob_death_proj"] * x["se"]) / x["prob_death"], axis=1
    )
    df.drop(columns=["prob_death"], inplace=True)
    df.rename(columns={"prob_death_proj": "prob_death"}, inplace=True)

    return df


def beta_year_optimizer(
    beta_year: float,
    life_table: pd.DataFrame,
    df_calibration: pd.DataFrame,
    sex: str,
    province: str, 
    year_initial: int,
    year: int,
) -> float:
    """Calculate the difference between the projected life expectancy and desired life expectancy.

    This function is passed to the ``scipy.optimize.brentq`` function. We want to find ``beta_year``
    such that the projected life expectancy is as close as possible to the desired life expectancy.
    
    Args:
        beta_year: The beta parameter for the given year.
        life_table: A dataframe containing the projected probability of death
            for the calibration year, for a given sex and province. Columns:

            * ``age``: the integer age.
            * ``sex``: one of ``M`` = male, ``F`` = female.
            * ``year``: the calibration calendar year.
            * ``province``: a 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``prob_death``: the probability of death for a given age, province, sex, and year.

        sex: one of ``M`` = male, ``F`` = female.
        province: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
            For all of Canada, set province to ``"CA"``.
        year_initial: The initial year with a known probability of death. This is the last year
            that the past data was collected.
        year: The current year.
    
    Returns:
        The difference between the projected life expectancy of the calibration year
        and the desired life expectancy.
    """
    desired_life_expectancies = df_calibration.loc[
        (df_calibration["sex"] == sex) &
        (df_calibration["province"] == province) &
        (df_calibration["projection_scenario"] == "M3")
    ]
    year = desired_life_expectancies["year"].values[-1]
    projected_life_table = get_projected_life_table_single_year(
        beta_year, life_table, year_initial, year, sex, province
    )
    logger.info(projected_life_table)

    life_expectancy = calculate_life_expectancy(projected_life_table)
    desired_life_expectancy = desired_life_expectancies.loc[
        desired_life_expectancies["year"] == year, "life_expectancy"
    ].values[0]

    return life_expectancy - desired_life_expectancy


def load_past_death_data() -> pd.DataFrame:
    """Load the past death data from the ``StatCan`` CSV file.
    
    Returns:
        A dataframe containing the probability of death and the standard error
        for each year, province, age, and sex.
        Columns:

        * ``year``: The integer calendar year.
        * ``province``: a 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``prob_death``: The probability that a person of the given age, sex, and province
          will die in the given year.
        * ``se``: The standard error of the probability of death.
    """ 
    logger.info("Loading mortality data from CSV file...")
    df = pd.read_csv(get_data_path("original_data/13100837.csv"))

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename the columns
    df.rename(
        columns={"REF_DATE": "year", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age"},
        inplace=True
    )

    # select the required columns
    df = df.loc[df["year"] >= STARTING_YEAR, ["year", "province", "sex", "age", "ELEMENT", "VALUE"]]

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
    df = pd.merge(df_prob, df_se, on=["year", "province", "sex", "age"], how="left")

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

        * ``year (int)``: The calendar year. Range ``[1988, 2073]``.
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
    return df


def get_projected_death_data(
    past_life_table: pd.DataFrame,
    df_calibration: pd.DataFrame,
    a: float = -0.03,
    b: float = -0.01,
    xtol: float = 0.00001
) -> pd.DataFrame:
    """Load the projected death data from ``StatCan`` CSV file.
    
    Args:
        past_life_table: A dataframe containing the probability of death and the standard error
            for each year, province, age, and sex. Columns:
            
            * ``year``: the integer calendar year.
            * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
              For all of Canada, set province to ``"CA"``.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``age``: the integer age.
            * ``prob_death``: the probability of death.
            * ``se``: the standard error of the probability of death.

        a: The lower bound for the beta parameter.
        b: The upper bound for the beta parameter.
        xtol: The tolerance for the beta parameter.
    
    Returns:
        A dataframe containing the predicted probability of death and the standard error
        for each year, province, age, and sex.
        Columns:

        * ``year``: The integer calendar year.
        * ``province``: A 2-letter string indicating the province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``prob_death``: The probability that a person of the given age, sex, and province
          will die in the given year.
        * ``se``: The standard error of the probability of death.
    """

    projected_life_table = pd.DataFrame({
        "year": np.array([], dtype=int),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "prob_death": [],
        "se": []
    })
    for province in past_life_table["province"].unique():
        life_table = past_life_table[past_life_table["province"] == province]
        starting_year = life_table["year"].max() + 1
        life_table = life_table[life_table["year"] == starting_year - 1]

        beta_year_female = optimize.brentq(
            beta_year_optimizer,
            a=a,
            b=b,
            args=(
                life_table,
                df_calibration,
                "F",
                province,
                starting_year - 1
            ),
            xtol=xtol
        )

        beta_year_male = optimize.brentq(
            beta_year_optimizer,
            a=a,
            b=b,
            args=(
                life_table,
                df_calibration,
                "M",
                province,
                starting_year - 1
            ),
            xtol=xtol
        )

        projected_life_table_province = pd.DataFrame({
            "year": np.array([], dtype=int),
            "province": [],
            "age": np.array([], dtype=int),
            "sex": [],
            "prob_death": [],
            "se": []
        })
        for year in range(starting_year, FINAL_YEAR + 1):
            # get the prob_death projections for the year and add to dataframe
            df_female = get_projected_life_table_single_year(
                beta_year_female, life_table, starting_year - 1, year, "F", province
            )
            df_male = get_projected_life_table_single_year(
                beta_year_male, life_table, starting_year - 1, year, "M", province
            )
            # combine the dataframes
            projected_life_table_single_year = pd.concat([df_female, df_male], axis=0)
            projected_life_table_province = pd.concat(
                [projected_life_table_province, projected_life_table_single_year],
                axis=0
            )

        projected_life_table = pd.concat(
            [projected_life_table, projected_life_table_province],
            axis=0
        )

    return projected_life_table



def generate_death_data():
    """Generate the mortality data CSV."""
    past_life_table = load_past_death_data()
    df_calibration = load_projected_death_data()
    projected_life_table = get_projected_death_data(past_life_table, df_calibration)
    life_table = pd.concat([past_life_table, projected_life_table], axis=0)

    # save the data
    file_path = get_data_path("processed_data/life_table.csv")
    logger.info(f"Saving data to {file_path}")
    life_table.to_csv(file_path, index=False)


if __name__ == "__main__":
    generate_death_data()