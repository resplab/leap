import pandas as pd
from leap.utils import get_data_path
from leap.logger import get_logger
from leap.data_generation.utils import format_age_group, get_province_id, get_sex_id
pd.options.mode.copy_on_write = True

logger = get_logger(__name__)

STARTING_YEAR = 1996
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


