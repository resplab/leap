import pandas as pd
import os
import pathlib
import datetime as dt
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from leap.utils import get_data_path, get_time_delta_tag, date_range, TimeDelta
from leap.data_generation.utils import get_province_id, get_sex_id, format_age_group, get_parser
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_TIMEPOINT = dt.datetime(1999, 1, 1)
MAX_TIMEPOINT = dt.datetime(2070, 1, 1)

# Most recent census date from StatCan; data switches from past to projected at this timepoint
CENSUS_TIMEPOINT = dt.datetime(2021, 1, 1)

# Time duration between data points in the original data from StatCan
TIME_DELTA_OD = TimeDelta(years=1)


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
    
def interpolate(
    data: pd.DataFrame,
    col_pred: str,
    formula: str,
    time_delta: TimeDelta
) -> pd.DataFrame:
    """Interpolate the values of a column for missing timepoints using a GLM.
    
    Args:
        data: The data to interpolate. Must contain a ``"timepoint"`` column.
        col_pred: The name of the column to predict.
        formula: The formula to use for the GLM, e.g. "timepoint + age".
        time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years, etc.

    Returns:
        A dataframe with the same columns as the input data, but with the values of the column to
        predict interpolated for the missing timepoints. The dataframe will contain rows for all
        timepoints between the minimum and maximum timepoints in the input data, with a step size of
        ``time_delta``.
    """

    initial_timepoint = data["timepoint"].min()
    final_timepoint = data["timepoint"].max()

    df_pred = pd.DataFrame(
        data=list(itertools.product(
            list(date_range(start=initial_timepoint, stop=final_timepoint + TIME_DELTA_OD, step=time_delta))
        )),
        columns=["timepoint"]
    )

    data["timepoint"] = data["timepoint"].apply(lambda x: (x - initial_timepoint).total_seconds())
    df_pred["timepoint"] = df_pred["timepoint"].apply(lambda x: (x - initial_timepoint).total_seconds())

    formula = f"{col_pred} ~ {formula}"
    model = smf.glm(formula=formula, data=data, family=sm.families.Gaussian())
    if data[col_pred].nunique() == 1:
        df_pred[f"{col_pred}_pred"] = data[col_pred].iloc[0]
    else:
        results = model.fit(maxiter=100)
        df_pred[f"{col_pred}_pred"] = results.predict(df_pred, which="mean", transform=True)
    
    df_pred["timepoint"] = df_pred["timepoint"].apply(lambda x: initial_timepoint + dt.timedelta(seconds=x))
    data["timepoint"] = data["timepoint"].apply(lambda x: initial_timepoint + dt.timedelta(seconds=x))
    df = pd.merge(df_pred, data, on=["timepoint"], how="left").sort_values(["timepoint"])
    for _, row in data.iterrows():
        timepoint = row["timepoint"]
        for col in [x for x in data.columns if x not in ["timepoint", col_pred]]:
            df.loc[
                (df["timepoint"] > timepoint) & 
                (df["timepoint"] < timepoint + TIME_DELTA_OD.to_dateoffset()), 
                col
            ] = row[col]
        df.loc[(df["timepoint"] == timepoint), f"{col_pred}_pred"] = row[col_pred]
    df.drop(columns=[col_pred], inplace=True)
    df.rename(columns={f"{col_pred}_pred": col_pred}, inplace=True)

    return df
    



def load_past_births_population_data(
    time_delta: TimeDelta,
    min_timepoint: dt.datetime = MIN_TIMEPOINT
) -> pd.DataFrame:
    """Load the past birth data from the CSV file.
    
    Args:
        time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years, etc.
        min_timepoint: The minimum timepoint to include in the data.

    Returns:
        The past birth data.
        Columns:
        
        * ``timepoint``: The date / time of the data.
        * ``province``: The 2-letter province ID.
        * ``N``: The total number of births in that time interval.
        * ``prop_male``: The proportion of births in that time interval that are male.
        * ``projection_scenario``: The projection scenario; all values are ``"past"``.
    """

    logger.info("Loading past population data from CSV file...")
    df = pd.read_csv(
        get_data_path("original_data/17100005.csv"),
        parse_dates=["REF_DATE"],
        low_memory=False
    )

    # select only the age = 0 age group and the timepoints >= min_timepoint
    df = df.loc[(df["REF_DATE"] >= min_timepoint) & (df["AGE_GROUP"] == "0 years")]
    df = df[["REF_DATE", "GEO", "SEX", "VALUE"]]
    df.rename(
        columns={"REF_DATE": "timepoint", "GEO": "province", "SEX": "sex", "VALUE": "N"},
        inplace=True
    )

    # convert province names to 2-letter province IDs
    df["province"] = df["province"].apply(get_province_id)

    # convert sex to 1-letter ID ("F", "M", "B")
    df["sex"] = df["sex"].apply(get_sex_id)

    # convert N to integer
    df["N"] = df["N"].apply(lambda x: int(x))

    # get the proportion male / female
    grouped_df = df.groupby(["timepoint", "province"])
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

    # Interpolate the birth estimates for the missing timepoints in the past data
    grouped_df = df.groupby(["province", "projection_scenario"])
    df = grouped_df.apply(lambda x: interpolate(
        data=x.reset_index(drop=True),
        col_pred="N",
        formula="timepoint",
        time_delta=time_delta
    )).reset_index(drop=True)
    df.sort_values(["province", "projection_scenario", "timepoint"], inplace=True)

    return df


def load_projected_births_population_data(
    min_timepoint: dt.datetime,
    max_timepoint: dt.datetime = MAX_TIMEPOINT
) -> pd.DataFrame:
    """Load the projected births data from the CSV file from ``StatCan``.

    Args:
        min_timepoint: The starting timepoint for the projected data.
        max_timepoint: The ending timepoint for the projected data.

    Returns:
        The projected births data.
        Columns:
        
        * ``timepoint``: The starting date / time of the time interval.
        * ``province``: The 2-letter province ID.
        * ``N``: The total number of births predicted for that time interval.
        * ``prop_male``: The proportion of predicted births in that time interval that are male.
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
    
    df = pd.read_csv(
        get_data_path("original_data/17100057.csv"),
        parse_dates=["REF_DATE"],
        low_memory=False
    )

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename columns
    df.rename(
        columns={
            "REF_DATE": "timepoint", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age",
            "VALUE": "N", "PROJECTION_SCENARIO": "projection_scenario"
        },
        inplace=True
    )

    # keep only rows where timepoint >= min_timepoint and age == "Under 1 year" (babies)
    df = df.loc[
        (df["timepoint"] >= min_timepoint) &
        (df["timepoint"] <= max_timepoint) &
        (df["age"] == "Under 1 year")
    ]

    # select columns
    df = df[["timepoint", "province", "projection_scenario", "sex", "age", "N"]]

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
    grouped_df = df.groupby(["timepoint", "province", "projection_scenario"])
    df["prop"] = grouped_df["N"].transform(lambda x: x / x.max())
    df["max_N"] = grouped_df["N"].transform(lambda x: x.max())

    # keep only male entries
    df = df.loc[df["sex"] == "M"]

    # drop N and sex columns
    df = df.drop(columns=["N", "sex", "age"])
    df.rename(columns={"max_N": "N", "prop": "prop_male"}, inplace=True)

    # Interpolate the birth estimates for the missing timepoints in the past data
    grouped_df = df.groupby(["province", "projection_scenario"])
    df = grouped_df.apply(lambda x: interpolate(
        data=x.reset_index(drop=True),
        col_pred="N",
        formula="timepoint",
        time_delta=time_delta
    )).reset_index(drop=True)
    df.sort_values(["province", "timepoint", "projection_scenario"], inplace=True)

    return df



def load_past_initial_population_data(
    time_delta: TimeDelta, min_timepoint: dt.datetime = MIN_TIMEPOINT
) -> pd.DataFrame:
    """Load the past initial population data from the CSV file.

    Args:
        time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years,
            1 month, etc.
        min_timepoint: The starting timepoint for the past data; only timepoints >= this value will
            be included in the returned data.
    
    Returns:
        The past initial population data.
        Columns:
        
        * ``timepoint``: The date / time of the data.
        * ``province``: The 2-letter province ID, e.g. ``BC``.
        * ``age``: The age of the population.
        * ``prop_male``: The proportion of the population in that age group that are male.
        * ``n_age``: The total number of people in that age group for the given time interval,
          province, and projection scenario.
        * ``n_birth``: The total number of births in the given time interval, province, and
          projection scenario.
        * ``prop``: The proportion of the total number of people in that age group
          to the total number of births in that time interval.
        * ``projection_scenario``: The projection scenario; all values are "past".
    """
    logger.info("Loading past population data from CSV file (StatCan 17100005)...")
    df = pd.read_csv(
        get_data_path("original_data/17100005.csv"),
        parse_dates=["REF_DATE"],
        low_memory=False
    )

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename the columns
    df.rename(
        columns={
            "REF_DATE": "timepoint", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age", "VALUE": "N"
        },
        inplace=True
    )

    # select the required columns
    df = df.loc[(df["timepoint"] > min_timepoint)][["timepoint", "province", "sex", "age", "N"]]

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

    # create a df to replace missing values with those of the next timepoint and age
    replacement_df = df.loc[
        (df["timepoint"].isin(missing_df["timepoint"] + TIME_DELTA_OD.to_dateoffset())) & 
        (df["age"].isin(missing_df["age"] + TIME_DELTA_OD.total_years()))
    ]
    replacement_df["age"] = replacement_df["age"] - TIME_DELTA_OD.total_years()
    replacement_df = replacement_df.drop(columns=["timepoint"])
    replacement_df.rename(columns={"N": "N_replace"}, inplace=True)

    # merge the two dfs
    replacement_df = pd.merge(missing_df, replacement_df, on=["sex", "age", "province"], how="left")

    # replace the missing values in the original df
    df = pd.merge(df, replacement_df, on=["sex", "age", "province", "timepoint"], how="left")
    df["N"] = df.apply(lambda x: x["N_replace"] if pd.isnull(x["N"]) else x["N"], axis=1)
    df = df.drop(columns=["N_replace"])

    # remove rows which are still missing values of N
    df = df.dropna(subset=["N"])

    # convert N to integer
    df["N"] = df["N"].apply(lambda x: int(x))

    # get the total population for a given time interval, province, and age
    grouped_df = df.groupby(["timepoint", "age", "province"])
    df["n_age"] = grouped_df["N"].transform(lambda x: x.sum())
    df["prop_male"] = df.apply(lambda x: x["N"] / x["n_age"] if x["n_age"] != 0 else 0, axis=1)

    # keep only male entries
    df = df.loc[df["sex"] == "M"]
    df.drop(columns=["sex", "N"], inplace=True)

    # interpolate
    grouped_df = df.groupby(["province", "age"])
    df = grouped_df.apply(lambda x: interpolate(
        data=x.reset_index(drop=True),
        col_pred="n_age",
        formula="timepoint",
        time_delta=time_delta
    )).reset_index(drop=True)

    # get the total number of births for a given time interval and province
    df_birth = df.loc[df["age"] == 0]
    df_birth["n_birth"] = df_birth["n_age"].values
    df_birth.drop(columns=["age", "n_age", "prop_male"], inplace=True)

    # add the births column to the main df
    df = pd.merge(df, df_birth, on=["province", "timepoint"], how="left")
    df["prop"] = df.apply(lambda x: 0 if x["n_birth"] == 0 else x["n_age"] / x["n_birth"], axis=1)



    # add projection_scenario column, all values = "past"
    df["projection_scenario"] = ["past"] * df.shape[0]
    df = df.sort_values(["province", "timepoint", "age"]).reset_index(drop=True)
    return df


def load_projected_initial_population_data(
    time_delta: TimeDelta,
    min_timepoint: dt.datetime,
    max_timepoint: dt.datetime = MAX_TIMEPOINT
) -> pd.DataFrame:
    """Load the projected initial population data from the CSV file.

    Args:
        time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years,
            1 month, etc.
        min_timepoint: The starting timepoint for the projected data.
        max_timepoint: The ending timepoint for the projected data.

    Returns:
        The projected initial population data.
        Columns:

        * ``timepoint``: The starting date / time of the time interval.
        * ``province``: The 2-letter province ID, e.g. ``BC``.
        * ``age``: The age of the population.
        * ``prop_male``: The proportion of the population in that age group that are male.
        * ``n_age``: The total number of people in that age group for the given time interval,
          province, and projection scenario.
        * ``n_birth``: The total number of births in the given time interval, province, and
          projection scenario.
        * ``prop``: The proportion of the total number of people in that age group to the total
          number of births in that time interval.
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
    df = pd.read_csv(
        get_data_path("original_data/17100057.csv"),
        parse_dates=["REF_DATE"],
        low_memory=False
    )

    # remove spaces from column names and make uppercase
    column_names = {}
    for column in df.columns:
        column_names[column] = column.upper().replace(" ", "_")
    df.rename(columns=column_names, inplace=True)

    # rename the columns
    df.rename(
        columns={
            "REF_DATE": "timepoint", "GEO": "province", "SEX": "sex", "AGE_GROUP": "age", "VALUE": "N",
            "PROJECTION_SCENARIO": "projection_scenario"
        },
        inplace=True
    )

    # select the required columns
    df = df.loc[(df["timepoint"] >= min_timepoint) & (df["timepoint"] <= max_timepoint)]
    df = df[["timepoint", "province", "sex", "age", "N", "projection_scenario"]]

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

    # get the total population for a given timepoint, province, age, and projection scenario
    grouped_df = df.groupby(["timepoint", "age", "province", "projection_scenario"])
    df["prop_male"] = grouped_df["N"].transform(lambda x: x / x.sum())
    df["n_age"] = grouped_df["N"].transform(lambda x: x.sum())

    # keep only male entries
    df = df.loc[df["sex"] == "M"]
    df.drop(columns=["sex", "N"], inplace=True)

    # interpolate
    grouped_df = df.groupby(["province", "age", "projection_scenario"])
    df = grouped_df.apply(lambda x: interpolate(
        data=x.reset_index(drop=True),
        col_pred="n_age",
        formula="timepoint",
        time_delta=time_delta
    )).reset_index(drop=True)

    # get the total number of births for a given timepoint, province, and projection scenario
    df_birth = df.loc[df["age"] == 0]
    df_birth["n_birth"] = df_birth["n_age"].values
    df_birth.drop(columns=["age", "n_age", "prop_male"], inplace=True)

    # add the births column to the main df
    df = pd.merge(df, df_birth, on=["province", "timepoint", "projection_scenario"], how="left")
    df["prop"] = df.apply(lambda x: x["n_age"] / x["n_birth"], axis=1)

    df = df.sort_values(["province", "timepoint", "age"]).reset_index(drop=True)
    return df


def generate_birth_estimate_data(time_delta: TimeDelta):
    """Create/update the ``birth_estimate.csv`` file.
    
    Args:
        time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years, etc.
    """
    past_population_data = load_past_births_population_data(time_delta)
    min_timepoint = past_population_data["timepoint"].max() + time_delta
    projected_population_data = load_projected_births_population_data(min_timepoint)
    birth_estimate = pd.concat([past_population_data, projected_population_data], axis=0)

    # Save the birth estimate data to a CSV file
    data_path = get_data_path(f"processed_data")
    time_delta_tag = get_time_delta_tag(time_delta)
    file_path = pathlib.Path(data_path, f"{time_delta_tag}/birth/birth_estimate.csv")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    logger.info(f"Saving data to {file_path}")
    birth_estimate.to_csv(file_path, index=False)


def generate_initial_population_data(time_delta: TimeDelta):
    """Create/update the ``initial_pop_distribution_prop.csv`` file.
    
    Args:
         time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years, etc.
    """
    past_population_data = load_past_initial_population_data(time_delta=time_delta)
    min_timepoint = past_population_data["timepoint"].max()
    projected_population_data = load_projected_initial_population_data(
        time_delta=time_delta, min_timepoint=min_timepoint
    )
    initial_population = pd.concat([past_population_data, projected_population_data], axis=0)

    # Save the initial population distribution data to a CSV file
    data_path = get_data_path(f"processed_data")
    time_delta_tag = get_time_delta_tag(time_delta)
    file_path = pathlib.Path(data_path, f"{time_delta_tag}/birth/initial_pop_distribution_prop.csv")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    logger.info(f"Saving data to {file_path}")
    initial_population.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    time_delta = TimeDelta(iso_string=args.time_delta)
    generate_initial_population_data(time_delta)
    generate_birth_estimate_data(time_delta)