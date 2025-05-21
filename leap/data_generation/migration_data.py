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
GENERATE = False
INTERPOLATE = True


def get_prev_year_population(
    df: pd.DataFrame, sex: str, year: int, age: int, min_year: int, min_age: int
) -> pd.Series:
    """Get the age, sex, probability of death, and population for the previous year.

    Args:
        df: A dataframe with the following columns:

            * ``year``: The calendar year.
            * ``sex``: One of ``M`` = male, ``F`` = female.
            * ``age``: The integer age.
            * ``N``: The population for a given year, age, sex, province, and projection scenario.
            * ``prob_death``: The probability that a person in the given year, age, sex,
              province, and projection scenario will die within the year.

        sex: One of ``F`` = female, ``M`` = male.
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
        n: The number of people living in Canada for a single age, sex, year, province, and
            projection scenario.
        n_prev: The number of people living in Canada in the previous year for the same
            age, sex, province, and projection scenario as defined for ``n``.
            So if ``n`` is the number of females aged ``10`` in the year
            ``2020``, ``n_prev`` is the number of females aged ``9`` in the year ``2019``.
        prob_death: The probability that a person with a given age and sex in a given
            year will die between the previous year and this year. So if the person is
            a female aged ``10`` in ``2020``, ``prob_death`` is the probability that a
            female aged ``9`` in ``2019`` will die by the age of ``10``.

    Returns:
        The change in population for a given year, age, and sex due to migration.
    """
    return n - n_prev * (1 - prob_death)


def get_n_migrants(delta_N: float) -> pd.Series:
    """Get the number of immigrants and emigrants in a single year for a given age and sex.

    .. important::
    
        **TODO**: This function is wrong. ``delta_N`` is the change in population due to migration.
        This function currently assumes that if ``delta_N < 0``, 100% of migration is
        emigration, and if ``delta_N > 0``, 100% of migration is immigration. This has
        led to the data being very inaccurate (for example, it appears as though people in their
        90s are emigrating a lot and people in their 20s are not). This will be remedied in a
        separate PR.

    Args:
        delta_N: The change in population for a given year, age, sex, province, and
            projection scenario due to migration.

    Returns:
        A ``pd.Series`` containing two values, the number of immigrants in a single year and the
        number of emigrants in a single year.
    """
    return pd.Series(
        [0 if delta_N < 0 else delta_N, 0 if delta_N > 0 else -delta_N],
        index=["n_immigrants", "n_emigrants",]
    )


def load_migration_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate migration data for the given provinces and years.
    
    Returns:
        A tuple containing two dataframes.
        The first dataframe contains the immigration data:
        
        * ``year``: The calendar year.
        * ``province``: A string indicating the 2-letter province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``projection_scenario``: The projection scenario.
        * ``n_immigrants``: The number of immigrants for a given year, province, sex, age, and
          projection scenario.
        * ``prop_immigrants_birth``: The proportion of immigrants for a given year, province,
          sex, age, and projection scenario, relative to the total number of births in that year
          for the given province and projection scenario.
        * ``prop_immigrants_year``: The proportion of immigrants for a given year, province,
          sex, age, and projection scenario, relative to the total number of immigrants in that
          year for the given province and projection scenario.

        The second dataframe contains the emigration data:

        * ``year``: The calendar year.
        * ``province``: A string indicating the 2-letter province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``projection_scenario``: The projection scenario.
        * ``n_emigrants``: The number of emigrants for a given year, province, sex, age, and
          projection scenario.
        * ``prop_emigrants_birth``: The proportion of emigrants for a given year, province,
          sex, age, and projection scenario, relative to the total number of births in that year
          for the given province and projection scenario.
        * ``prop_emigrants_year``: The proportion of emigrants for a given year, province,
          sex, age, and projection scenario, relative to the total number of emigrants in that
          year for the given province and projection scenario.

    """
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

        # Select only the data for the given province and the years after the starting year
        df = df_population.loc[
            (df_population["year"] >= STARTING_YEAR) & 
            (df_population["province"] == province)
        ]
        df = df[["year", "age", "province", "n_age", "prop_male", "projection_scenario"]]

        # Get the total number of M / F for each year, age, and projection scenario
        df_male = df.copy()
        df_female = df.copy()
        df_male["N"] = df_male.apply(
            lambda x: int(x["n_age"] * x["prop_male"]), axis=1
        )
        df_male["sex"] = ["M"] * df_male.shape[0]
        df_female["N"] = df_female.apply(
            lambda x: int(x["n_age"] * (1 - x["prop_male"])), axis=1
        )
        df_female["sex"] = ["F"] * df_female.shape[0]
        df = pd.concat([df_male, df_female], axis=0)
        df.drop(columns=["n_age", "prop_male"], inplace=True)

        # Get the list of projection scenarios, excluding "past"
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
            df_birth["n_birth"] = grouped_df.transform("sum")["N"]
            df_birth = df_birth.loc[df_birth["sex"] == "F", ["year", "n_birth"]]

            # get the next year and age for each entry
            df_proj[["year_prev", "age_prev", "n_prev", "prob_death_prev"]] = df_proj.apply(
                lambda x: get_prev_year_population(
                    df_proj, x["sex"], x["year"], x["age"], min_year, min_age
                ),
                axis=1
            )

            # remove the missing data
            df_proj = df_proj.dropna(subset=["n_prev"])

            # compute the change in population due to migration, delta_N
            df_proj["delta_N"] = df_proj.apply(
                lambda x: get_delta_n(x["N"], x["n_prev"], x["prob_death_prev"]), axis=1
            )

            # add the n_birth column to df_proj
            df_proj = pd.merge(df_proj, df_birth, on="year", how="left")

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
    
    return df_immigration, df_emigration


def generate_migration_data():
    df_immigration, df_emigration = load_migration_data()
    file_path = get_data_path("processed_data/migration/immigration_table.csv")
    logger.info(f"Saving data to {file_path}")
    df_immigration.to_csv(file_path, index=False)

    file_path = get_data_path("processed_data/migration/emigration_table.csv")
    logger.info(f"Saving data to {file_path}")
    df_emigration.to_csv(file_path, index=False)

def interpolate_emigration_data(method: str = "linear"):
    """
    Interpolates the ``migration/emigration_table.csv`` by months between the years.
    
    Args:
        method: The interpolation method to use (linear or loess).
    """
    # Check for valid method
    if method not in ["linear", "loess"]:
        raise ValueError(f"method was {method}. Must be one of ['linear', 'loess']")
    
    # Load dataset
    logger.info("Loading processed population data from emigration CSV file...")
    df = pd.read_csv(
        get_data_path("processed_data/migration/emigration_table.csv")
    )
    
    # Sort values
    df = df.sort_values(["age", "province", "proj_scenario", "year"])

    # Define columns to interpolate
    interp_cols = ["F", "M"]

    # Storage for interpolated output
    all_rows = []

    # Grouping
    group_cols = ["age", "province", "proj_scenario"]
    for group_key, group_df in df.groupby(group_cols):
        logger.info(f"Interpolating emigration data for group {group_key}.")
        group_df = group_df.sort_values("year")
        
        for i in range(len(group_df) - 1):
            row_start = group_df.iloc[i]
            row_end   = group_df.iloc[i + 1]
            
            for m in range(12):  # 12 points between years
                fraction = m / 12
                year_interp = row_start["year"] + fraction
                
                interpolated_row = {
                    "year_float": year_interp,
                    "age": row_start["age"],
                    "province": row_start["province"],
                    "proj_scenario": row_start["proj_scenario"],
                }
                for col in interp_cols:
                    interpolated_row[col] = (
                        row_start[col] + fraction * (row_end[col] - row_start[col])
                    )
                all_rows.append(interpolated_row)

    # Add the final year point for each group
    for group_key, group_df in df.groupby(group_cols):
        final_row = group_df.loc[group_df["year"].idxmax()]
        all_rows.append({
            "year_float": final_row["year"],
            "age": final_row["age"],
            "province": final_row["province"],
            "proj_scenario": final_row["proj_scenario"],
            "F": final_row["F"],
            "M": final_row["M"],
        })

    # Convert to DataFrame and sort
    monthly_df = pd.DataFrame(all_rows)
    monthly_df = monthly_df.sort_values(["age", "province", "proj_scenario", "year_float"])
    
    # Convert year_float to year-month string like "YYYY-MM"
    monthly_df["year_month"] = monthly_df["year_float"].apply(
        lambda y: f"{int(y):04d}-{min(12, round((y - int(y)) * 12) + 1):02d}"
    )
    
    # Drop year_float
    monthly_df = monthly_df.drop(columns="year_float")

    # Move year_month to the front
    monthly_df = monthly_df[
        ["year_month"] + [col for col in monthly_df.columns if col != "year_month"]
    ]

    # Save to CSV
    file_path = get_data_path(
        "processed_data/migration/emigration_table_monthly.csv",
        create=True
    )
    logger.info(f"Saving data to {file_path}")
    monthly_df.to_csv(file_path, float_format="%.8g", index=False)

if __name__ == "__main__":
    if GENERATE:
        generate_migration_data()
        
    if INTERPOLATE:
        interpolate_emigration_data(method="linear")
