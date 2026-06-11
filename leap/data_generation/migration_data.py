import pandas as pd
import numpy as np
from leap.utils import get_data_path
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_TIMEPOINT = 2000
MIN_TIMEPOINT_PROJ = 2021
MAX_TIMEPOINT = 2065
PROVINCES = ["CA", "BC"]


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


def load_migration_data() -> pd.DataFrame:
    """Generate migration data for the given provinces and years.

    Returns:
        A dataframe with the following columns:

        * ``year``: The calendar year.
        * ``province``: A string indicating the 2-letter province abbreviation, e.g. ``"BC"``.
          For all of Canada, set province to ``"CA"``.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: The integer age.
        * ``projection_scenario``: The projection scenario.
        * ``delta_n``: The signed change in population for a given year, age, sex, province, and
          projection scenario due to net migration. Positive values indicate net immigration;
          negative values indicate net emigration.
        * ``prop_migrants_birth``: The signed proportion of ``delta_n`` relative to the total
          number of births in that year for the given province and projection scenario.
          Positive = net immigration, negative = net emigration.
        * ``prop_immigrants_year``: For cells where ``delta_n > 0``, the proportion of immigrants
          for this age and sex relative to the total number of immigrants in that year. Zero for
          emigration cells. Denominator includes only immigration cells.
        * ``prop_emigrants_year``: For cells where ``delta_n < 0``, the proportion of emigrants
          for this age and sex relative to the total number of emigrants in that year. Zero for
          immigration cells. Denominator includes only emigration cells.
        * ``prob_emigration``: For cells where ``delta_n < 0``, the per-person probability of
          emigrating (``abs(delta_n) / N``). Zero for immigration cells.

    """
    logger.info("Loading initial population data from CSV file...")
    df_population = pd.read_csv(
        get_data_path("processed_data/birth/initial_population.csv")
    )
    logger.info("Loading mortality data from CSV file...")
    life_table = pd.read_csv(get_data_path("processed_data/life_table.csv"))

    df_migration = pd.DataFrame({
        "year": np.array([], dtype=int),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "projection_scenario": [],
        "delta_n": [],
        "prop_migrants_birth": [],
        "prop_immigrants_year": [],
        "prop_emigrants_year": [],
        "prob_emigration": []
    })

    for province in PROVINCES:
        logger.info(f"Processing migration data for province {province}...")

        # Select only the data for the given province and the years after the starting year
        df = df_population.loc[
            (df_population["year"] >= MIN_TIMEPOINT) &
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
                ~((df["projection_scenario"] == "past") & (df["year"] == MIN_TIMEPOINT_PROJ))
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

            # get the previous year's cohort for each entry
            df_proj[["year_prev", "age_prev", "n_prev", "prob_death_prev"]] = df_proj.apply(
                lambda x: get_prev_year_population(
                    df_proj, x["sex"], x["year"], x["age"], min_year, min_age
                ),
                axis=1
            )

            # remove the missing data
            df_proj = df_proj.dropna(subset=["n_prev"])

            # compute the signed population change due to net migration
            df_proj["delta_n"] = df_proj.apply(
                lambda x: get_delta_n(x["N"], x["n_prev"], x["prob_death_prev"]), axis=1
            )

            # add the n_birth column to df_proj
            df_proj = pd.merge(df_proj, df_birth, on="year", how="left")

            # signed proportion relative to births
            df_proj["prop_migrants_birth"] = df_proj["delta_n"] / df_proj["n_birth"]

            # year proportions with separate denominators for immigration and emigration
            df_proj["n_immigrants_year"] = df_proj.groupby("year")["delta_n"].transform(
                lambda x: x.clip(lower=0).sum()
            )
            df_proj["n_emigrants_year"] = df_proj.groupby("year")["delta_n"].transform(
                lambda x: (-x).clip(lower=0).sum()
            )
            df_proj["prop_immigrants_year"] = df_proj.apply(
                lambda x: x["delta_n"] / x["n_immigrants_year"]
                    if x["delta_n"] > 0 and x["n_immigrants_year"] > 0 else 0.0,
                axis=1
            )
            df_proj["prop_emigrants_year"] = df_proj.apply(
                lambda x: -x["delta_n"] / x["n_emigrants_year"]
                    if x["delta_n"] < 0 and x["n_emigrants_year"] > 0 else 0.0,
                axis=1
            )

            # per-person probability of emigrating
            df_proj["prob_emigration"] = df_proj.apply(
                lambda x: -x["delta_n"] / x["N"] if x["delta_n"] < 0 and x["N"] > 0 else 0.0,
                axis=1
            )

            df_migration_proj = df_proj[[
                "year", "province", "age", "sex", "projection_scenario",
                "delta_n", "prop_migrants_birth",
                "prop_immigrants_year", "prop_emigrants_year", "prob_emigration"
            ]].copy()

            # convert the "past" projection scenario to the given projection scenario
            df_migration_proj["projection_scenario"] = projection_scenario

            df_migration = pd.concat([df_migration, df_migration_proj], axis=0)

    return df_migration


def generate_migration_data():
    df_migration = load_migration_data()
    file_path = get_data_path("processed_data/migration") / "migration_table.csv"
    logger.info(f"Saving data to {file_path}")
    df_migration.to_csv(file_path, index=False)


if __name__ == "__main__":
    generate_migration_data()
