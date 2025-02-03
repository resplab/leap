import pandas as pd
import numpy as np
from typing import Tuple
from leap.utils import get_data_path
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 2000
STARTING_YEAR_PROJ = 2021
MAX_YEAR = 2065
PROVINCES = ["CA", "BC"]


def get_prev_year_population(
    df: pd.DataFrame, sex: str, year: int, age: int, min_year: int, min_age: int
) -> pd.Series:
    """Get the age, sex, probability of death, and population for the previous year.

    Args:
        df: TODO.
        sex: One of "F" = female, "M" = male.
        year: The calendar year.
        age: The integer age.
        min_year: The minimum year in the dataframe.
        min_age: The minimum age in the dataframe.

    Returns:
        The age, sex, probability of death, and population for the previous year.
    """
    if year == min_year or age == min_age:
        return pd.Series(
            [np.nan, np.nan, np.nan, np.nan],
            index=["year_prev", "age_prev", "n_prev", "prob_death_prev"]
        )
    else:
        return df.loc[
            (df["sex"] == sex) & 
            (df["year"] == year - 1) & 
            (df["age"] == age - 1)
        ][["year", "age", "N", "prob_death"]].iloc[0].rename({
            "year": "year_prev",
            "age": "age_prev",
            "N": "n_prev",
            "prob_death": "prob_death_prev"
        })
    

def get_delta_n(n: float, n_prev: float, prob_death: float) -> float:
    """Get the population change due to migration for a given age and sex in a single year.

    Args:
        n: The number of people living in Canada for a single age, sex, and year.
        n_prev: The number of people living in Canada for the same sex as ``n``, in the
            previous year and age. So if ``n`` is the number of females aged ``10`` in the year
            ``2020``, ``n_prev`` is the number of females aged ``9`` in the year ``2019``.
        prob_death: The probability that a person with a given age and sex in a given
            year will die between the previous year and this year. So if the person is
            a female aged ``10`` in ``2020``, ``prob_death`` is the probability that a
            female aged ``9`` in ``2019`` will die by the age of ``10``.

    Returns:
        The change in population for a given year, age, and sex due to migration.
    """
    return n - n_prev * (1 - prob_death)


def get_n_migrants(delta_N: float) -> Tuple[float, float]:
    """Get the number of immigrants and emigrants in a single year for a given age and sex.

    TODO: This function is wrong. delta_N is the change in population due to migration. This
        function currently assumes that if delta_N is less than zero, 100% of migration is
        emigration, and if it is greater than zero, 100% of migration is immigration. This has
        led to the data being very inaccurate (for example, it appears as though people in their
        90s are emigrating a lot and people in their 20s are not). This will be remedied in a
        separate PR.

    Args:
        delta_N: The change in population for a given year, age, and sex due to migration.

    Returns:
        A vector containing two values, the number of immigrants in a single year and the number
        of emigrants in a single year.
    """
    return (0 if delta_N < 0 else delta_N, 0 if delta_N > 0 else -delta_N)


def load_migration_data():
    logger.info("Loading initial population data from CSV file...")
    df_population = pd.read_csv(
        get_data_path("processed_data/birth/initial_pop_distribution_prop.csv")
    )
    logger.info("Loading mortality data from CSV file...")
    life_table = pd.read_csv(get_data_path("processed_data/life_table.csv"))
    
    df_immigration = pd.DataFrame({
        "year": np.array([], dtype=int),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "projection_scenario": [],
        "n_immigrants": [],
        "prop_immigrants_birth": [],
        "prop_immigrants_year": []
    })

    df_emigration = pd.DataFrame({
        "year": np.array([], dtype=int),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "projection_scenario": [],
        "n_emigrants": [],
        "prop_emigrants_birth": [],
        "prop_emigrants_year": []
    })

    for province in PROVINCES:
        logger.info(f"Processing migration data for province {province}...")
        df = df_population.loc[
            (df_population["year"] >= STARTING_YEAR) & 
            (df_population["province"] == province)
        ]
        df = df[["year", "age", "province", "n_age", "prop_male", "projection_scenario"]]

        df_male = df.copy()
        df_female = df.copy()
        df_male["N"] = df_male.apply(
            lambda x: x["n_age"] * x["prop_male"], axis=1
        )
        df_female["N"] = df_female.apply(
            lambda x: x["n_age"] * (1 - x["prop_male"]), axis=1
        )
        df = pd.concat([df_male, df_female], axis=0)
        df.drop(columns=["n_age", "prop_male"], inplace=True)

        projection_scenarios = df.loc[df["projection_scenario"] != "past", "projection_scenario"].unique()
        min_year = df["year"].min()
        min_age = 0

        for projection_scenario in projection_scenarios:
            logger.info(f"Projection scenario: {projection_scenario}")

            # select only the current projection scenario and the past projection scenario
            df_proj = df.loc[
                (df["projection_scenario"].isin(["past", projection_scenario])) & 
                ~((df["projection_scenario"] == "past") & (df["year"] == STARTING_YEAR_PROJ))
            ]

            # join to the life table to get death probabilities
            df_proj = df_proj.merge(
                life_table, on=["year", "age", "province", "sex"], how="left"
            )

            # get the number of births in each year
            df_birth = df_proj.loc[df_proj["age"] == 0]
            grouped_df = df_birth.groupby("year")
            df_birth["n_birth"] = grouped_df["N"].sum().reset_index()
            df_birth = df_birth.loc[df_birth["sex"] == "F", ["year", "n_birth"]]

            # get the next year and age for each entry
            df_proj[["year_prev", "age_prev", "n_prev", "prob_death_prev"]] = df_proj.apply(
                lambda x: get_prev_year_population(df_proj, x["sex"], x["year"], x["age"], min_year, min_age),
                axis=1
            )

            # remove the missing data
            df_proj = df_proj.dropna(subset=["n_prev"])

            # compute the change in population
            df_proj["delta_N"] = df_proj.apply(
                lambda x: get_delta_n(x["N"], x["n_prev"], x["prob_death"]), axis=1
            )

            # add the n_birth column to df_proj
            df_proj = df_proj.merge(df_birth, on="year", how="left")

            # get the number of immigrants/emigrants
            df_migration_proj = df_proj.copy()
            df_migration_proj[["n_immigrants", "n_emigrants"]] = df_migration_proj["delta_N"].apply(
                lambda x: get_n_migrants(x)
            )

            # compute the proportion of immigrants/emigrants to the number of births in a year
            df_migration_proj["prop_immigrants_birth"] = df_migration_proj.apply(
                lambda x: x["n_immigrants"] / x["n_birth"], axis=1
            )
            df_migration_proj["prop_emigrants_birth"] = df_migration_proj.apply(
                lambda x: x["n_emigrants"] / x["n_birth"], axis=1
            )

            df_migration_proj = df_migration_proj[[
                "year", "province", "age", "sex", "projection_scenario",
                "prop_immigrants_birth", "prop_emigrants_birth", "n_immigrants", "n_emigrants"
            ]]

            # get the migrants for a given age and sex relative to the migrants for that year
            grouped_df = df_migration_proj.groupby("year")
            df_migration_proj["n_immigrants_year"] = grouped_df["n_immigrants"].transform("sum")
            df_migration_proj["n_emigrants_year"] = grouped_df["n_emigrants"].transform("sum")
            df_migration_proj["prop_immigrants_year"] = df_migration_proj.apply(
                lambda x: x["n_immigrants"] / x["n_immigrants_year"], axis=1
            )
            df_migration_proj["prop_emigrants_year"] = df_migration_proj.apply(
                lambda x: x["n_emigrants"] / x["n_emigrants_year"], axis=1
            )

            # remove n_immigrants_year, n_emigrants_year
            df_migration_proj.drop(columns=["n_immigrants_year", "n_emigrants_year"], inplace=True)

            # convert the "past" projection scenario to the given projection scenario
            df_migration_proj["projection_scenario"] = [projection_scenario] * df_migration_proj.shape[0]

            # create separate immigration and emigration dataframes
            df_immigration_proj = df_migration_proj.drop(
                columns=["n_emigrants", "prop_emigrants_year", "prop_emigrants_birth"]
            )
            df_emigration_proj = df_migration_proj.drop(
                columns=["n_immigrants", "prop_immigrants_year", "prop_immigrants_birth"]
            )

            # append the immigration and emigration dataframes for the current projection scenario
            df_immigration = pd.concat([df_immigration, df_immigration_proj], axis=0)
            df_emigration = pd.concat([df_emigration, df_emigration_proj], axis=0)
    
    file_path = get_data_path("processed_data/migration/master_immigration_table.csv")
    logger.info(f"Saving data to {file_path}")
    df_immigration.to_csv(file_path, index=False)

    file_path = get_data_path("processed_data/migration/master_emigration_table.csv")
    logger.info(f"Saving data to {file_path}")
    df_emigration.to_csv(file_path, index=False)


if __name__ == "__main__":
    load_migration_data()
