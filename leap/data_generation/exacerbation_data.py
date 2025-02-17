import pandas as pd
import numpy as np
import re
from leap.control import Control
from leap.utils import get_data_path, Sex
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


# We assume that asthma diagnoses are made at age 3 and older
MIN_AGE = 3
# The CIHI data only goes up to age 90
MAX_AGE = 90

PROVINCES = ["BC", "CA"]
MAX_YEARS = {
    "BC": 2043,
    "CA": 2065
}

# Probability of a very severe exacerbation:
# The number of exacerbations per year per person with asthma.
# Very severe exacerbations are defined as exacerbations that require hospitalization.
# Source: Symbicort Given as Needed in Mild Asthma (SYGMA) II study (Bateman et al., 2018).
PROB_HOSP = 0.026

CONTROL_PARAMETERS = {
    "β0": 0.0,
    "βage": 3.543038,
    "βsex": 0.234780,
    "βsexage": -0.8161495,
    "βsexage2": -1.1654264,
    "βage2": -3.4980710,
    "θ": [-1e5, -0.3950, 2.754, 1e5]
}
BETA_CONTROL = [0.1880058, 0.3760116, 0.5640174]


def exacerbation_prediction(
    sex: str, age: int, beta_control: list[float] | None = None
):
    """TODO.
    
    Args:
        sex: One of "M" or "F".
        age: Integer age, a value in ``[3, 90]``.
        beta_control: A list of three floats, the control parameters.

    Returns:
        The predicted number of exacerbations per year per person with asthma.
    """

    if beta_control is None:
        beta_control = BETA_CONTROL
    if age < 3:
        return 0
    else:
        control = Control(parameters=CONTROL_PARAMETERS, hyperparameters=None)
        control_levels = control.compute_control_levels(sex=Sex(sex), age=age)
        return np.exp(np.sum(control_levels.as_array() * np.log(beta_control)))


def parse_sex(x: str) -> str | float:
    """Reformat a string containing sex information.
    
    Args:
        x: A string containing sex information. For example, ``Female``, ``Male``, ``M``, or ``F``.
        
    Returns:
        Either ``M`` or ``F``, or ``np.nan`` if the string does not contain sex information.
    """
    if "M" in x:
        return "M"
    elif "F" in x:
        return "F"
    else:
        return np.nan
    

def parse_age(x: str) -> int | float:
    """Reformat a string containing age information.
    
    Args:
        x: A string containing age information. If the string is in the format
            ``{sex}_{age}``, we parse the integer age. For example, ``F_90`` or ``M_1``.
            
    Returns:
        The integer age, or ``np.nan`` if the string does not contain age information.
    """
    regex = re.compile(r"^([MF])_[0-9]{1,2}$")
    if regex.match(x) is not None:
        return int(x.split("_")[1])
    else:
        return np.nan
    

def load_hospitalization_data(
    province: str = "CA", starting_year: int = 2000, min_age: int = 3
) -> pd.DataFrame:
    """Load the hospitalization data for the given province and starting year.

    The data is from the ``Hospital Morbidity Database (HMDB)`` from the
    ``Canadian Institute for Health Information (CIHI)``:
    https://www.cihi.ca/en/hospital-morbidity-database-hmdb-metadata

    The hospitalization data was collected from patients presenting to a hospital in Canada
    due to an asthma exacerbation. We will use this data to calibrate the exacerbation model.
    
    Args:
        province: The province for which to load the hospitalization data.
        starting_year: The starting year for which to load the hospitalization data.
        min_age: The minimum age for to be used in the data. We are assuming that
            asthma diagnoses are made at age 3 and older, so the default is 3.
        
    Returns:
        A dataframe with the following columns:
        * year: The year of the data.
        * sex: One of "M" or "F".
        * age: Integer age, a value in ``[3, 90]``.
        * hospitalization_rate: The observed number of hospitalizations relative to the number
          of people in a given year, age, and sex.
    """

    # Load the hospitalization data
    df = pd.read_csv(get_data_path(f"original_data/asthma_hosp/{province}/tab1_rate.csv"))
    df.rename(columns={"fiscal_year": "year"}, inplace=True)
    df = df[df["year"] >= starting_year]

    # Convert the columns M, F, N, M_1, M_2, etc to single column "type"
    df = df.melt(id_vars=["year"], var_name="type", value_name="hospitalization_rate")
    df["type"] = df.apply(
        lambda x: x["type"].replace("+", ""), axis=1
    )

    # Remove NA values from the hospitalization_rate column
    df = df.dropna(subset=["hospitalization_rate"])

    # Remove "+" from the type column
    df["type"] = df.apply(
        lambda x: x["type"].replace("+", ""), axis=1
    )

    # Parse sex from the type column
    df["sex"] = df["type"].apply(
        lambda x: parse_sex(x)
    )

    # Remove rows with sex = NA
    df = df.dropna(subset=["sex"])

    # Parse age from the type column
    df["age"] = df["type"].apply(
        lambda x: parse_age(x)
    )

    # Remove rows with age = NA
    df = df.dropna(subset=["age"])
    df["age"] = df["age"].astype(int)

    # Drop the type column
    df.drop(columns=["type"], inplace=True)

    # Filter out age < 3
    df = df.loc[df["age"] >= min_age]

    # Sort by year, sex, and age
    df = df.sort_values(by=["year", "sex", "age"])

    return df


def load_population_data(
    province: str,
    starting_year: int,
    projection_scenario: str,
    max_year: int,
    min_age: int = 3,
    max_age: int = 90
) -> pd.DataFrame:
    """Load the population data for the given province, starting year, and projection scenario.
    
    The population data was generated by the ``leap/data_generation/birth_data.py`` script.
    
    Args:
        province: The 2-letter abbreviation for the province.
        starting_year: The starting year for the population data.
        projection_scenario: The projection scenario for the population data.
        max_year: The maximum year for the population data.
        min_age: The minimum age for the population data.
        max_age: The maximum age for the population data.
    
    Returns:
        A dataframe with the following columns:

        * year: The year of the data.
        * age: A value in ``[min_age, max_age]``.
        * province: The 2-letter province abbreviation.
        * sex: One of "M" or "F".
        * n: The number of people in a given year, age, sex, province, and projection scenario.

    """

    df = pd.read_csv(
        get_data_path("processed_data/birth/initial_pop_distribution_prop.csv")
    )

    # Filter the population data by province and starting year
    df = df.loc[
        (df["province"] == province) & (df["year"] >= starting_year)
    ]

    # Add a sex column and compute n_age for each sex (n_age_sex)
    df_male = df.copy()
    df_male["sex"] = ["M"] * df.shape[0]
    df_male["n_age_sex"] = df.apply(
        lambda x: x["n_age"] * x["prop_male"], axis=1
    )
    df_female = df.copy()
    df_female["sex"] = ["F"] * df.shape[0]
    df_female["n_age_sex"] = df.apply(
        lambda x: x["n_age"] * (1 - x["prop_male"]), axis=1
    )
    df = pd.concat([df_female, df_male])
    df["n_age_sex"] = df["n_age_sex"].astype(int)

    # Filter the population data by projection scenario and year
    df = df.loc[
        (df["projection_scenario"] == "past") & (df["year"] <= 2021)
        | (
            (df["projection_scenario"] == projection_scenario)
            & (df["year"] > 2021)
        )
    ]

    # Remove unnecessary columns
    df.drop(
        columns=["projection_scenario", "n_age", "prop_male", "prop", "n_birth"],
        inplace=True
    )
    df.rename(columns={"n_age_sex": "n"}, inplace=True)


    df = df.loc[df["year"] <= max_year]
    df = df.loc[df["age"] >= min_age]

    # Set any age > max_age to the max_age
    df["age"] = df["age"].apply(
        lambda x: min(x, max_age)
    )

    # Sum the max_age rows to a single value of n
    grouped_df = df.groupby(["year", "sex", "age"])
    df["n"] = grouped_df["n"].transform(lambda x: sum(x))
    df.drop_duplicates(inplace=True)

    return df


def exacerbation_calibrator(
    province: str = "CA",
    starting_year: int = 2000,
    max_year: int = 2065,
    min_age: int = MIN_AGE,
    max_age: int = MAX_AGE,
    prob_hosp: float = PROB_HOSP,
    projection_scenario: str = "M3"
):
    """TODO.
    
    Args:
        province: The 2-letter abbreviation for the province.
        starting_year: The starting year for the calibration.
        max_year: The maximum year for the calibration.
        min_age: The minimum age for the calibration.
        max_age: The maximum age for the calibration.
        prob_hosp: The probability of a very severe exacerbation, defined as an
            exacerbation that requires hospitalization.
        projection_scenario: The projection scenario for the population data. One of:

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

            See: `StatCan Projection Scenarios
            <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
    """

    df_prev_inc = pd.read_csv(get_data_path("processed_data/master_asthma_prev_inc.csv"))
    df_prev = df_prev_inc[["year", "age", "sex", "prev"]]
    df_prev["sex"] = df_prev.apply(
        lambda x: "F" if x["sex"]==0 else "M", axis=1
    )

    # Canada Institute for Health Information (CIHI) data on hospitalizations due to asthma
    df_cihi = load_hospitalization_data(province, starting_year, min_age)

    final_year = max(df_cihi["year"])
    future_years = list(range(final_year + 1, max_year + 1))
    
    # Append a copy of the final year data for each of the future years, changing the year column
    for year in future_years:
        df_cihi_year = df_cihi[df_cihi["year"] == final_year].copy()
        df_cihi_year["year"] = [year] * df_cihi_year.shape[0]
        df_cihi = pd.concat([df_cihi, df_cihi_year])

    # Load population data
    df_population = load_population_data(
        province, starting_year, projection_scenario, max_year, min_age, max_age
    )

    # Calculate the number of hospitalizations per 100 000 people
    # hospitalization_rate: The observed number of hospitalizations relative to the number
    # of people in a given year, age, and sex.
    # n: The number of people in a given year, age, and sex.
    df_target = pd.merge(df_population, df_cihi, on=["year", "sex", "age"], how="left")
    df_target["n_hosp_per_100000"] = df_target.apply(
        lambda x: x["hospitalization_rate"] * x["n"] / 100000, axis=1
    )

    # Calculate the number of people with asthma per 100 people
    # prev: The prevalence of asthma in a given year, age, and sex per 100 people.
    # n: The number of people in a given year, age, and sex.
    df_target = pd.merge(df_target, df_prev, on=["year", "sex", "age"], how="left")
    df_target["n_asthma"] = df_target.apply(
        lambda x: x["prev"] * x["n"], axis=1
    )

    df_target["mean_annual_exacerbation"] = df_target.apply(
        lambda x: exacerbation_prediction(x["sex"], x["age"]), axis=1
    )

    # Calculate the predicted number of exacerbations per 100 000 people
    # mean_annual_exacerbation: The mean number of exacerbations per 1000 people.
    # n_asthma: The number of people with asthma for a given age, sex, and year.
    df_target["n_exacerbations_per_100000_pred"] = df_target.apply(
        lambda x: x["mean_annual_exacerbation"] * x["n_asthma"], axis=1
    )

    # Calculate the predicted number of hospitalizations per 100 000 people
    # prob_hosp: the number of exacerbations per year per person with asthma.
    # n_exacerbations_per_100000_pred: the predicted number of exacerbations per 100 000 people.
    df_target["n_hosp_per_100000_pred"] = df_target.apply(
        lambda x: prob_hosp * x["n_exacerbations_per_100000_pred"], axis=1
    )

    # Calculate the ratio between the observed and predicted number of hospitalizations
    # per 100 000 people
    df_target["calibrator_multiplier"] = df_target.apply(
        lambda x: x["n_hosp_per_100000"] / x["n_hosp_per_100000_pred"], axis=1
    )

    # Drop unnecessary columns
    df = df_target[["year", "sex", "age", "calibrator_multiplier"]]

    # Add province column
    df["province"] = [province] * df.shape[0]

    return df



def generate_exacerbation_calibration():
    df = pd.DataFrame({
        "year": np.array([], dtype=int),
        "sex": [],
        "age": np.array([], dtype=int),
        "calibrator_multiplier": []
    })
    for province in PROVINCES:
        df_province = exacerbation_calibrator(province, max_year=MAX_YEARS[province])
        df = pd.concat([df, df_province], axis=0)

    df.to_csv(get_data_path("processed_data/master_calibrated_exac.csv"), index=False)


if __name__ == "__main__":
    generate_exacerbation_calibration()
