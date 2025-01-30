import pandas as pd
import pathlib
from leap.utils import get_data_path
from leap.data_generation.utils import get_province_id, get_sex_id, format_age_group
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 1999


def get_projection_scenario_id(projection_scenario: str) -> str:
    return projection_scenario.replace("Projection scenario ", "")[0:2]


def filter_age_group(age_group: str) -> bool:
    if "100" in age_group:
        return True
    else:
        return (
            "to" not in age_group and "over" not in age_group
            and "All" not in age_group and "Median" not in age_group
            and "Average" not in age_group
        )


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
    df.sort_values(["province", "year", "projection_scenario"], inplace=True)

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
    df.sort_values(["province", "year", "projection_scenario"], inplace=True)

    return df



def load_past_initial_population_data() -> pd.DataFrame:
    logger.info("Loading past population data from CSV file...")
    df = pd.read_csv(get_data_path("original_data/17100005.csv"))

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename the columns
    df.rename(
        columns={"REF_DATE": "year", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age", "VALUE": "N"},
        inplace=True
    )

    # select the required columns
    df = df.loc[(df["year"] >= STARTING_YEAR + 1)][["year", "province", "sex", "age", "N"]]

    # remove grouped categories such as "Median", "Average", "All" and format age as integer
    df = df.loc[df["age"].apply(filter_age_group)]
    df["age"] = df["age"].apply(format_age_group)

    # convert province names to 2-letter province IDs
    df["province"] = df["province"].apply(get_province_id)

    # convert sex to 1-letter ID ("F", "M", "B")
    df["sex"] = df["sex"].apply(get_sex_id)

    # remove sex category "Both"
    df = df.loc[df["sex"] != "B"]

    # find the missing values of N
    missing_df = df.loc[df["N"].isnull()]
    missing_df = missing_df.drop(columns=["N"])

    # create a df to replace missing values with those of the next year and age
    replacement_df = df.loc[
        (df["year"].isin(missing_df["year"] + 1)) & (df["age"].isin(missing_df["age"] + 1))
    ]
    replacement_df["age"] = replacement_df["age"] - 1
    replacement_df = replacement_df.drop(columns=["year"])
    replacement_df.rename(columns={"N": "N_replace"}, inplace=True)

    # merge the two dfs
    replacement_df = pd.merge(missing_df, replacement_df, on=["sex", "age", "province"], how="left")

    # replace the missing values in the original df
    df = pd.merge(df, replacement_df, on=["sex", "age", "province", "year"], how="left")
    df["N"] = df.apply(lambda x: x["N_replace"] if pd.isnull(x["N"]) else x["N"], axis=1)
    df = df.drop(columns=["N_replace"])

    # remove rows which are still missing values of N
    df = df.dropna(subset=["N"])

    # convert N to integer
    df["N"] = df["N"].apply(lambda x: int(x))

    # get the total population for a given year, province, and age
    grouped_df = df.groupby(["year", "age", "province"])
    df["prop_male"] = grouped_df["N"].transform(lambda x: x / x.sum())
    df["n_age"] = grouped_df["N"].transform(lambda x: x.sum())

    # get the total number of births for a given year and province
    df_birth = df.loc[df["age"] == 0]
    df_birth["N_birth"] = df_birth["n_age"].values
    df_birth.drop(columns=["age", "N", "n_age", "prop_male"], inplace=True)

    # add the births column to the main df
    df = pd.merge(df, df_birth, on=["province", "sex", "year"], how="left")
    df["prop"] = df["N"] / df["N_birth"]

    # keep only male entries
    df = df.loc[df["sex"] == "M"]
    df.drop(columns=["sex", "N"], inplace=True)

    # add projection_scenario column, all values = "past"
    df["projection_scenario"] = ["past"] * df.shape[0]
    df = df.sort_values(["province", "year", "age"]).reset_index(drop=True)
    return df


def load_projected_initial_population_data(min_year: int) -> pd.DataFrame:
    logger.info("Loading projected population data from CSV file...")
    df = pd.read_csv(get_data_path("original_data/17100057.csv"))

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename the columns
    df.rename(
        columns={
            "REF_DATE": "year", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age", "VALUE": "N",
            "PROJECTION_SCENARIO": "projection_scenario"
        },
        inplace=True
    )

    # select the required columns
    df = df.loc[(df["year"] >= min_year)]
    df = df[["year", "province", "sex", "age", "N", "projection_scenario"]]

    # convert the long form of the projection scenario to the 2-letter ID
    df["projection_scenario"] = df["projection_scenario"].apply(get_projection_scenario_id)

    # remove grouped categories such as "Median", "Average", "All" and format age as integer
    df = df.loc[df["age"].apply(filter_age_group)]
    df["age"] = df["age"].apply(format_age_group)

    # convert province names to 2-letter province IDs
    df["province"] = df["province"].apply(get_province_id)

    # convert sex to 1-letter ID ("F", "M", "B")
    df["sex"] = df["sex"].apply(get_sex_id)

    # remove sex category "Both"
    df = df.loc[df["sex"] != "B"]

    # remove rows which are missing values of N
    df = df.dropna(subset=["N"])

    # multiply the :N column by 1000 and convert to integer
    df["N"] = df["N"].apply(lambda x: int(round(x * 1000, 0)))

    # get the total population for a given year, province, age, and projection scenario
    grouped_df = df.groupby(["year", "age", "province", "projection_scenario"])
    df["prop_male"] = grouped_df["N"].transform(lambda x: x / x.sum())
    df["n_age"] = grouped_df["N"].transform(lambda x: x.sum())

    # get the total number of births for a given year, province, and projection scenario
    df_birth = df.loc[df["age"] == 0]
    df_birth["N_birth"] = df_birth["n_age"].values
    df_birth.drop(columns=["age", "N", "n_age", "prop_male"], inplace=True)

    # add the births column to the main df
    df = pd.merge(df, df_birth, on=["province", "sex", "year", "projection_scenario"], how="left")
    df["prop"] = df["N"] / df["N_birth"]

    # keep only male entries
    df = df.loc[df["sex"] == "M"]
    df.drop(columns=["sex", "N"], inplace=True)

    df = df.sort_values(["province", "year", "age"]).reset_index(drop=True)
    return df


def generate_birth_estimate_data():
    past_population_data = load_past_population_data()
    min_year = past_population_data["year"].max() + 1
    projected_population_data = load_projected_population_data(min_year)
    birth_estimate = pd.concat([past_population_data, projected_population_data], axis=0)
    file_path = get_data_path("processed_data/master_birth_estimate.csv")
    logger.info(f"Saving data to {file_path}")
    birth_estimate.to_csv(file_path, index=False)


def generate_initial_population_data():
    past_population_data = load_past_initial_population_data()
    min_year = past_population_data["year"].max()
    projected_population_data = load_projected_initial_population_data(min_year)
    initial_population = pd.concat([past_population_data, projected_population_data], axis=0)
    file_path = get_data_path("processed_data/master_initial_pop_distribution_prop.csv")
    logger.info(f"Saving data to {file_path}")
    initial_population.to_csv(file_path, index=False)


if __name__ == "__main__":
    generate_initial_population_data()
    generate_birth_estimate_data()