import pandas as pd
import pathlib
from leap.utils import get_data_path
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__)

DATA_PATH = get_data_path("original_data")
STARTING_YEAR = 1999


PROVINCE_MAP = {
    "Canada": "CA",
    "British Columbia": "BC",
    "Alberta": "AB",
    "Saskatchewan": "SK",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
    "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Prince Edward Island": "PE",
    "Yukon": "YT",
    "Northwest Territories": "NT",
    "Nunavut": "NU"
}


def get_province_id(province: str) -> str:
    return PROVINCE_MAP[province]


def get_sex_id(sex: str) -> str:
    return sex[0:1]


def get_projection_scenario_id(projection_scenario: str) -> str:
    return projection_scenario.replace("Projection scenario ", "")[0:2]



def load_past_population_data() -> pd.DataFrame:
    logger.info("Loading past population data from CSV file...")
    df = pd.read_csv(get_data_path("original_data/17100005.csv"))

    df = df.loc[(df["REF_DATE"] >= STARTING_YEAR) & (df["AGE_GROUP"] == "0 years")][["REF_DATE", "GEO", "SEX", "VALUE"]]
    df.rename(
        columns={"REF_DATE": "year", "GEO": "province", "SEX": "sex", "VALUE": "N"},
        inplace=True
    )

    # convert province names to 2-letter province IDs
    df["province"] = df["province"].apply(get_province_id)

    # convert sex to 1-letter ID ("F", "M", "B")
    df["sex"] = df["sex"].apply(get_sex_id)

    # convert N to integer
    df["N"] = df["N"].apply(lambda x: int(x))

    # get the proportion male / female
    grouped_df = df.groupby(["year", "province"])
    df["prop"] = grouped_df["N"].transform(lambda x: x / x.max())
    df["max_N"] = grouped_df["N"].transform(lambda x: x.max())

    # keep only male entries
    df = df.loc[df["sex"] == "M"]

    # drop N and sex columns
    df = df.drop(columns=["N", "sex"])

    # rename max_N to N and prop to prop_male
    df.rename(columns={"max_N": "N", "prop": "prop_male"}, inplace=True)

    # add :projection_scenario column, all values = "past"
    df["projection_scenario"] = ["past"] * df.shape[0]

    return df


def load_projected_population_data(min_year: int) -> pd.DataFrame:
    logger.info("Loading projected population data from CSV file...")
    
    df = pd.read_csv(get_data_path("original_data/17100057.csv"))

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # keep only rows where REF_DATE >= min_year and AGE_GROUP == "Under 1 year" (babies)
    df = df.loc[
        (df["REF_DATE"] >= min_year) & 
        (df["AGE_GROUP"] == "Under 1 year")
    ]

    # select columns
    df = df[["REF_DATE", "GEO", "PROJECTION_SCENARIO", "SEX", "AGE_GROUP", "VALUE"]]

    # rename columns
    df.rename(
        columns={
            "REF_DATE": "year", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age",
            "VALUE": "N", "PROJECTION_SCENARIO": "projection_scenario"
        },
        inplace=True
    )

    # convert the long form of the projection scenario to the 2-letter ID
    df["projection_scenario"] = df["projection_scenario"].apply(get_projection_scenario_id)

    # convert province names to 2-letter province IDs
    df["province"] = df["province"].apply(get_province_id)

    # convert sex to 1-letter ID ("F", "M", "B")
    df["sex"] = df["sex"].apply(get_sex_id)

    # format the age group string
    df["age"] = [0] * df.shape[0]

    # remove rows which are missing values of N
    df = df.dropna(subset=["N"])

    # multiply the N column by 1000 and convert to integer
    df["N"] = df["N"].apply(lambda x: int(round(x * 1000, 0)))

    # get the proportion male / female
    grouped_df = df.groupby(["year", "province", "projection_scenario"])
    df["prop"] = grouped_df["N"].transform(lambda x: x / x.max())
    df["max_N"] = grouped_df["N"].transform(lambda x: x.max())

    # keep only male entries
    df = df.loc[df["sex"] == "M"]

    # drop N and sex columns
    df = df.drop(columns=["N", "sex", "age"])
    df.rename(columns={"max_N": "N", "prop": "prop_male"}, inplace=True)

    return df


def process_birth_estimate_data():
    past_population_data = load_past_population_data()
    min_year = past_population_data["year"].max() + 1
    projected_population_data = load_projected_population_data(min_year)
    birth_estimate = pd.concat([past_population_data, projected_population_data], axis=0)
    file_path = get_data_path("processed_data/master_birth_estimate.csv")
    logger.info(f"Saving data to {file_path}")
    birth_estimate.to_csv(file_path, index=False)