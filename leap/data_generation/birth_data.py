import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from leap.utils import get_data_path
from leap.data_generation.utils import (get_province_id,
                                        get_sex_id, format_age_group,
                                        interpolate_years_to_months)
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 1999
GENERATE = False
INTERPOLATE = True

def get_projection_scenario_id(projection_scenario: str) -> str:
    """Convert the long form of the projection scenario to the 2-letter ID.
    
    Args:
        projection_scenario: The long form of the projection scenario, e.g.
            ``Projection scenario M1``.
        
    Returns:
        The 2-letter ID of the projection scenario, e.g. ``M1``.
    """
    return projection_scenario.replace("Projection scenario ", "")[0:2]


def filter_age_group(age_group: str) -> bool:
    """Filter out grouped categories such as "Median", "Average", "All", "to", "over".

    Args:
        age_group: The age group string.

    Returns:
        ``True`` if the age group is not a grouped category, ``False`` otherwise.
    """
    FILTER_WORDS = ["Median", "Average", "All", "to", "over"]
    if "100" in age_group:
        return True
    else:
        return not any(word in age_group for word in FILTER_WORDS)


def load_past_births_population_data() -> pd.DataFrame:
    """Load the past birth data from the CSV file.
    
    Returns:
        The past birth data.
        Columns:
        
        * ``year``: The year of the data.
        * ``province``: The 2-letter province ID.
        * ``N``: The total number of births in that year.
        * ``prop_male``: The proportion of births in that year that are male.
        * ``projection_scenario``: The projection scenario; all values are ``"past"``.
    """

    logger.info("Loading past population data from CSV file...")
    df = pd.read_csv(get_data_path("original_data/17100005.csv"))

    # select only the age = 0 age group and the years >= STARTING_YEAR
    df = df.loc[(df["REF_DATE"] >= STARTING_YEAR) & (df["AGE_GROUP"] == "0 years")]
    df = df[["REF_DATE", "GEO", "SEX", "VALUE"]]
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

    # add projection_scenario column, all values = "past"
    df["projection_scenario"] = ["past"] * df.shape[0]
    df.sort_values(["province", "year", "projection_scenario"], inplace=True)

    return df


def load_projected_births_population_data(min_year: int) -> pd.DataFrame:
    """Load the projected births data from the CSV file from ``StatCan``.

    Args:
        min_year: The starting year for the projected data.
    
    Returns:
        The projected births data.
        Columns:
        
        * ``year``: The year of the data.
        * ``province``: The 2-letter province ID.
        * ``N``: The total number of births predicted for that year.
        * ``prop_male``: The proportion of predicted births in that year that are male.
        * ``projection_scenario``: The projection scenario, one of:

          * ``LG``: low-growth projection
          * ``HG``: high-growth projection
          * ``M1``: medium-growth 1 projection
          * ``M2``: medium-growth 2 projection
          * ``M3``: medium-growth 3 projection
          * ``M4``: medium-growth 4 projection
          * ``M5``: medium-growth 5 projection
          * ``M6``: medium-growth 6 projection
          * ``FA``: fast-aging projection
          * ``SA``: slow-aging projection

    """
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
    """Load the past initial population data from the CSV file.
    
    Returns:
        The past initial population data.
        Columns:
        
        * ``year``: The calendar year.
        * ``province``: The 2-letter province ID, e.g. ``BC``.
        * ``age``: The age of the population.
        * ``prop_male``: The proportion of the population in that age group that are male.
        * ``n_age``: The total number of people in that age group for the given year, province, and
          projection scenario.
        * ``n_birth``: The total number of births in the given year, province, and
          projection scenario.
        * ``prop``: The proportion of the total number of people in that age group
          to the total number of births in that year.
        * ``projection_scenario``: The projection scenario; all values are "past".
    """
    logger.info("Loading past population data from CSV file...")
    df = pd.read_csv(get_data_path("original_data/17100005.csv"))

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename the columns
    df.rename(
        columns={
            "REF_DATE": "year", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age", "VALUE": "N"
        },
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
    df_birth["n_birth"] = df_birth["n_age"].values
    df_birth.drop(columns=["age", "N", "n_age", "prop_male"], inplace=True)

    # add the births column to the main df
    df = pd.merge(df, df_birth, on=["province", "sex", "year"], how="left")
    df["prop"] = df.apply(lambda x: x["n_age"] / x["n_birth"], axis=1)

    # keep only male entries
    df = df.loc[df["sex"] == "M"]
    df.drop(columns=["sex", "N"], inplace=True)

    # add projection_scenario column, all values = "past"
    df["projection_scenario"] = ["past"] * df.shape[0]
    df = df.sort_values(["province", "year", "age"]).reset_index(drop=True)
    return df


def load_projected_initial_population_data(min_year: int) -> pd.DataFrame:
    """Load the projected initial population data from the CSV file.

    Args:
        min_year: The starting year for the projected data.

    Returns:
        The projected initial population data.
        Columns:

        * ``year``: The calendar year.
        * ``province``: The 2-letter province ID, e.g. ``BC``.
        * ``age``: The age of the population.
        * ``prop_male``: The proportion of the population in that age group that are male.
        * ``n_age``: The total number of people in that age group for the given year, province, and
          projection scenario.
        * ``n_birth``: The total number of births in the given year, province, and
          projection scenario.
        * ``prop``: The proportion of the total number of people in that age group to the total
          number of births in that year.
        * ``projection_scenario``: The projection scenario, one of:

          * ``LG``: low-growth projection
          * ``HG``: high-growth projection
          * ``M1``: medium-growth 1 projection
          * ``M2``: medium-growth 2 projection
          * ``M3``: medium-growth 3 projection
          * ``M4``: medium-growth 4 projection
          * ``M5``: medium-growth 5 projection
          * ``M6``: medium-growth 6 projection
          * ``FA``: fast-aging projection
          * ``SA``: slow-aging projection
    """

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
    df_birth["n_birth"] = df_birth["n_age"].values
    df_birth.drop(columns=["age", "N", "n_age", "prop_male"], inplace=True)

    # add the births column to the main df
    df = pd.merge(df, df_birth, on=["province", "sex", "year", "projection_scenario"], how="left")
    df["prop"] = df.apply(lambda x: x["n_age"] / x["n_birth"], axis=1)

    # keep only male entries
    df = df.loc[df["sex"] == "M"]
    df.drop(columns=["sex", "N"], inplace=True)

    df = df.sort_values(["province", "year", "age"]).reset_index(drop=True)
    return df


def generate_birth_estimate_data():
    """Create/update the ``birth_estimate.csv`` file."""
    past_population_data = load_past_births_population_data()
    min_year = past_population_data["year"].max() + 1
    projected_population_data = load_projected_births_population_data(min_year)
    birth_estimate = pd.concat([past_population_data, projected_population_data], axis=0)
    file_path = get_data_path("processed_data/birth/birth_estimate.csv")
    logger.info(f"Saving data to {file_path}")
    birth_estimate.to_csv(file_path, index=False)


def generate_initial_population_data():
    """Create/update the ``initial_pop_distribution_prop.csv`` file."""
    past_population_data = load_past_initial_population_data()
    min_year = past_population_data["year"].max()
    projected_population_data = load_projected_initial_population_data(min_year)
    initial_population = pd.concat([past_population_data, projected_population_data], axis=0)
    file_path = get_data_path("processed_data/birth/initial_pop_distribution_prop.csv")
    logger.info(f"Saving data to {file_path}")
    initial_population.to_csv(file_path, index=False)


def interpolate_population_data(method: str = "linear"):
    """
    Interpolates the ``birth/initial_pop_distribution_prop.csv`` by months between the years.
    
    The target values are:
        - ``n_age``
        - ``n_birth``
        - ``prop``
        - ``prop_male``
    Each target value is grouped by:
        - ``age``
        - ``province``
        - ``projection_scenario``
    
    Args:
        method: The interpolation method to use (linear or loess).
    """
    # Check for valid method
    if method not in ["linear", "loess"]:
        raise ValueError(f"method was {method}. Must be one of ['linear', 'loess']")
    
    # Load dataset
    logger.info("Loading processed population data from pop distribution CSV file...")
    df = pd.read_csv(
        get_data_path("processed_data/birth/initial_pop_distribution_prop.csv")
    )

    # Define columns to interpolate
    interp_cols = ["n_age", "n_birth", "prop", "prop_male"]

    # Storage for interpolated output
    all_rows = []

    # Grouping
    group_cols = ["age", "province", "projection_scenario"]
    for group_key, group_df in df.groupby(group_cols):
        logger.info(f"Interpolating population data with {method} for group {group_key}.")
        group_df = group_df.sort_values("year")
        
        # Create monthly x-values for interpolation
        year_min = group_df["year"].min()
        year_max = group_df["year"].max()
        monthly_x = np.linspace(year_min, year_max, int((year_max - year_min) * 12) + 1)
        
        if method == "linear":
            for i in range(len(group_df) - 1):
                row_start = group_df.iloc[i]
                row_end = group_df.iloc[i + 1]
                
                row_age = row_start["age"]
                row_province = row_start["province"]
                row_scenario = row_start["projection_scenario"]
                
                for m in range(12):
                    fraction = m / 12
                    year_interp = row_start["year"] + fraction

                    interpolated_row = {
                        "year_float": year_interp,
                        "age": row_age,
                        "province": row_province,
                        "projection_scenario": row_scenario,
                    }
                    for col in interp_cols:
                        interpolated_row[col] = (
                            row_start[col] + fraction * (row_end[col] - row_start[col])
                        )
                    all_rows.append(interpolated_row)

            # Add the final year (constant values of last year)
            final_row = group_df.loc[group_df["year"].idxmax()]
            all_rows.append({
                "year_float": final_row["year"],
                "age": final_row["age"],
                "province": final_row["province"],
                "projection_scenario": final_row["projection_scenario"],
                **{col: final_row[col] for col in interp_cols}
            })
        elif method == "loess":
            for col in interp_cols:
                smoothed = lowess(
                    endog=group_df[col],   # y value
                    exog=group_df["year"], # x value
                    frac=0.5,              # controls smoothing
                    return_sorted=True
                )
                interp_values = np.interp(monthly_x, smoothed[:, 0], smoothed[:, 1])
                
                # Store into temp dict by column
                if col == interp_cols[0]:
                    rows = [{
                        "year_float": x,
                        "age": group_key[0],
                        "province": group_key[1],
                        "projection_scenario": group_key[2],
                        col: val
                    } for x, val in zip(monthly_x, interp_values)]
                else:
                    for r, val in zip(rows, interp_values):
                        r[col] = val
            all_rows.extend(rows)

    # Convert to DataFrame and sort
    monthly_df = pd.DataFrame(all_rows)
    monthly_df = monthly_df.sort_values(["province", "projection_scenario", "year_float", "age"])
    
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
        "processed_data/birth/initial_pop_distribution_prop_monthly.csv",
        create=True
    )
    logger.info(f"Saving {method}-interpolated data to {file_path}")
    monthly_df.to_csv(file_path, float_format="%.8g", index=False)


def interpolate_birth_estimate_data(method: str = "linear"):
    """
    Interpolates the ``birth/birth_estimate.csv`` by months between the years
    """
    # Check for valid method
    if method not in ["linear", "loess"]:
        raise ValueError(f"method was {method}. Must be one of ['linear', 'loess']")
    
    # Load dataset
    logger.info("Loading processed population data from birth estimate CSV file...")
    df = pd.read_csv(
        get_data_path("processed_data/birth/birth_estimate.csv")
    )

    # Define target columns to interpolate
    interp_cols = ["N", "prop_male"]

    # Storage for interpolated output
    all_rows = []

    # Grouping
    group_cols = ["province", "projection_scenario"]
    
    for group_key, group_df in df.groupby(group_cols):
        logger.info(f"Interpolating birth data with {method} for group {group_key}.")
        group_df = group_df.sort_values("year")
        
        # Create monthly x-values for interpolation
        year_min = group_df["year"].min()
        year_max = group_df["year"].max()
        monthly_x = np.linspace(year_min, year_max, int((year_max - year_min) * 12) + 1)
        
        if method == "linear":
            for i in range(len(group_df) - 1):
                row_start = group_df.iloc[i]
                row_end = group_df.iloc[i + 1]
                
                # cache constant values within row
                row_province = row_start["province"]
                row_scenario = row_start["projection_scenario"]
                
                for m in range(12):
                    fraction = m / 12
                    year_interp = row_start["year"] + fraction

                    interpolated_row = {
                        "year_float": year_interp,
                        "province": row_province,
                        "projection_scenario": row_scenario,
                    }
                    for col in interp_cols:
                        interpolated_row[col] = (
                            row_start[col] + fraction * (row_end[col] - row_start[col])
                        )
                    all_rows.append(interpolated_row)

            # Add the final year (same as before)
            final_row = group_df.loc[group_df["year"].idxmax()]
            all_rows.append({
                "year_float": final_row["year"],
                "province": final_row["province"],
                "projection_scenario": final_row["projection_scenario"],
                **{col: final_row[col] for col in interp_cols}
            })
        elif method == "loess":
            for col in interp_cols:
                smoothed = lowess(
                    endog=group_df[col],   # y value
                    exog=group_df["year"], # x value
                    frac=0.5,              # controls smoothing
                    return_sorted=True
                )
                interp_values = np.interp(monthly_x, smoothed[:, 0], smoothed[:, 1])
                
                # Store into temp dict by column
                if col == interp_cols[0]: # built row once for first column
                    rows = [{
                        "year_float": x,
                        "province": group_key[0],
                        "projection_scenario": group_key[1],
                        col: val
                    } for x, val in zip(monthly_x, interp_values)]
                else: # reuse row for next columns
                    for r, val in zip(rows, interp_values):
                        r[col] = val
            all_rows.extend(rows)

    # Convert to DataFrame and sort
    monthly_df = pd.DataFrame(all_rows)
    monthly_df = monthly_df.sort_values(["province", "projection_scenario", "year_float"])
    
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
    file_path = get_data_path("processed_data/birth/birth_estimate_monthly.csv", create=True)
    logger.info(f"Saving data to {file_path}")
    monthly_df.to_csv(file_path, float_format="%.8g", index=False)
   

if __name__ == "__main__":
    if GENERATE:
        generate_initial_population_data()
        generate_birth_estimate_data()
    
    if INTERPOLATE:
        # Interpolate initial population data
        # interpolate_years_to_months(
        #     dataset="processed_data/birth/initial_pop_distribution_prop.csv",
        #     group_cols=["age", "province", "projection_scenario"],
        #     interp_cols=["n_age", "n_birth", "prop", "prop_male"],
        #     method="linear"
        # )
        # Interpolate birth estimate data
        interpolate_years_to_months(
            dataset="processed_data/birth/birth_estimate.csv",
            group_cols=["province", "projection_scenario"],
            interp_cols=["N", "prop_male"],
            method="linear"
        )