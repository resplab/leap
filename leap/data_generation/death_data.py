import pandas as pd
from leap.utils import get_data_path
from leap.logger import get_logger
from leap.data_generation.utils import format_age_group, get_province_id, get_sex_id
pd.options.mode.copy_on_write = True

logger = get_logger(__name__)

STARTING_YEAR = 1996
def calculate_life_expectancy(life_table: pd.DataFrame) -> float:
    """Determine the life expectancy for a person born in a given year.

    The life expectancy can be calculated from the death probability using the formulae
    delineated here: https://www.ssa.gov/oact/HistEst/CohLifeTables/LifeTableDefinitions.pdf.
    
    Args:
        life_table: A dataframe containing the probability of death for a single year,
            province and sex, for each age. Columns:
                * ``age``: the integer age.
                * ``sex``: one of "M" or "F".
                * ``year``: the integer calendar year.
                * ``province``: A string indicating the province abbreviation, e.g. "BC".
                  For all of Canada, set province to "CA".
                * ``prob_death``: the probability of death for a given age, province, sex, and year.
    Returns:
        The life expectancy for a person born in the given year, in a given province,
        for a given sex.
    """

    df = life_table.sort_values("age").copy()
    df.set_index("age", inplace=True)
    n_alive_by_age_0 = 100000
    n_alive_by_age = []
    for age in df.index:
        if age == 0:
            n_alive_by_age.append(n_alive_by_age_0)
        else:
            n_alive_by_age.append(
                n_alive_by_age[age - 1] * (1 - df.loc[age - 1, "prob_death"])
            )

    df["n_alive_by_age"] = n_alive_by_age
    df["n_person_years_interval"] = df.apply(
        lambda x: x["n_alive_by_age"] - 0.5 * x["prob_death"] * x["n_alive_by_age"], axis=1
    )

    df.loc[0, "n_person_years_interval"] = (
        df.loc[1, "n_person_years_interval"] +
        0.1 * df.loc[0, "prob_death"] * n_alive_by_age_0
    )

    df.loc[110, "n_person_years_interval"] = df.loc[110, "n_alive_by_age"] * 1.4

    df["n_person_years_after_age"] = df["n_person_years_interval"].sort_index(ascending=False).cumsum().sort_index()

    df["n_years_left_at_age"] = df.apply(
        lambda x: x["n_person_years_after_age"] / x["n_alive_by_age"], axis=1
    )

    life_expectancy = df.loc[0, "n_years_left_at_age"]

    return life_expectancy



def get_prob_death_projected(
    prob_death: float, year_index: int, beta_year: float
) -> float:
    """Given the prob death for a past year, calculate the prob death in a future year.

    Args:
        prob_death: The probability of death for the initial year (determined by past data).
        year_index: The number of years between the current year and the initial year.
            For example, if our initial year is 2020, and we want to compute the probability of
            death in 2028, the ``year_index`` would be 8.
        beta_year: The beta parameter for the given year.

    Returns:
        The projected probability of death for the current year.
    """
    prob_death = min(prob_death, 0.9999999999)
    odds = (prob_death / (1 - prob_death)) * np.exp(year_index * beta_year)
    prob_death_projected = max(min(odds / (1 + odds), 1), 0)
    return prob_death_projected



def get_projected_life_table_single_year(
    beta_year: float, life_table: pd.DataFrame, starting_year: int, year_index: int,
    sex: str, province: str
) -> pd.DataFrame:
    """Get the life table for a single year.

    Args:
        beta_year: The beta parameter for the given year.
        life_table: A dataframe containing the projected probability of death
            for the starting year, for a given sex and province. Columns:

            - ``age``: the integer age.
            - ``sex``: one of "M" or "F".
            - ``year``: the starting calendar year.
            - ``province``: a string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".
            - ``prob_death``: the probability of death for a given age, province, sex, and year.

        starting_year: The calendar year when the projections begin.
        year_index: The number of years between the current year and the starting year.
            For example, if our initial year is 2020, and we want to compute the probability of
            death in 2028, the ``year_index`` would be 8.
        sex: one of "M" or "F".
        province: a string indicating the province abbreviation, e.g. "BC".
            For all of Canada, set province to "CA".

    Returns:
        A dataframe containing the projected probability of death for the given year,
        sex, and province.
    """
    df = life_table.loc[(life_table["sex"] == sex) & (life_table["province"] == province)].copy()
    df["prob_death_proj"] = df["prob_death"].apply(
        lambda x: get_prob_death_projected(x, year_index, beta_year)
    )

    df["year"] = [starting_year + year_index - 1] * df.shape[0]

    df["se"] = df.apply(
        lambda x: (x["prob_death_proj"] * x["se"]) / x["prob_death"], axis=1
    )
    df.drop(columns=["prob_death"], inplace=True)
    df.rename(columns={"prob_death_proj": "prob_death"}, inplace=True)

    return df
def load_past_death_data() -> pd.DataFrame:
    """Load the past death data from the StatCan CSV file.
    
    Returns:
        A dataframe containing the probability of death and the standard error
        for each year, province, age, and sex. Columns:
            * ``year``: the integer calendar year.
            * ``province``: a string indicating the province abbreviation, e.g. "BC".
            For all of Canada, set province to "CA".
            * ``sex``: one of "M" or "F".
            * ``age``: the integer age.
            * ``prob_death``: the probability of death.
            * ``se``: the standard error of the probability of death.
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


