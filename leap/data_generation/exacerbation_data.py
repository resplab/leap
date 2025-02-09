import pandas as pd
import numpy as np
import re
from scipy.stats import logistic
from leap.utils import get_data_path
from leap.logger import get_logger
pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_AGE = 3
MAX_AGE = 90
BETA_CONTROL = [0.1880058, 0.3760116, 0.5640174]
THETA = [-1e5, -0.3950, 2.754, 1e5]
PROVINCES = ["BC", "CA"]

# we need: 
# 1) asthma prev
# 2) target rate (this is in per pop)
# 3) population 
# 4) 2) + 3) => annual number of severe hospitalizations needed
# 5) 1) + exacerbation module => annual number of exacerbations
# 6) so we can identify a tuner to do this

# asthma prev


def control_prediction(sex: int, age: int, theta: list | None = None):
    if theta is None:
        theta = THETA

    age_scaled = age / 100
    eta = (
        age_scaled * 3.543038 + sex * 0.234780 + age_scaled * sex * -0.8161495 +
        age_scaled ** 2 * sex * -1.1654264 + age_scaled ** 2 * -3.4980710
    )
    control_probabilities = []
    for i in range(len(theta) - 1):
        control_probabilities.append(
            logistic.cdf(theta[i + 1] - eta) - logistic.cdf(theta[i] - eta)
        )
    return control_probabilities


def exacerbation_prediction(
    sex: str, age: int, beta_control: list[float] | None = None
):
    """"""
    if beta_control is None:
        beta_control = BETA_CONTROL
    if age < 3:
        return 0
    else:
        if sex == "M":
            control = control_prediction(1, age)
        else:
            control = control_prediction(0, age)
        return np.exp(np.sum(control * np.log(beta_control)))


def parse_sex(x: str):
    if "M" in x:
        return "M"
    elif "F" in x:
        return "F"
    else:
        return np.nan
    

def parse_age(x: str):
    regex = re.compile(r"^([MF])_[0-9]{1,2}$")
    if regex.match(x) is not None:
        return int(x.split("_")[1])
    else:
        return np.nan
    

def load_hospitalization_data(province: str = "CA", starting_year: int = 2000) -> pd.DataFrame:
    """Load the hospitalization data for the given province and starting year.

    The data is from the ``Hospital Morbidity Database (HMDB)`` from the
    ``Canadian Institute for Health Information (CIHI)``:
    https://www.cihi.ca/en/hospital-morbidity-database-hmdb-metadata

    The hospitalization data was collected from patients presenting to a hospital in Canada
    due to an asthma exacerbation. We will use this data to calibrate the exacerbation model.
    
    Args:
        province (str): The province for which to load the hospitalization data.
        starting_year (int): The starting year for which to load the hospitalization data.
        
    Returns:
        A dataframe with the following columns:
        * year: The year of the data.
        * sex: One of "M" or "F".
        * age: Integer age, a value in ``[3, 90]``.
        * true_rate: The true rate of hospitalization for the given year, age, and sex.
    """

    # Load the hospitalization data
    df = pd.read_csv(get_data_path(f"processed_data/asthma_hosp/{province}/tab1_rate.csv"))
    df.rename(columns={"fiscal_year": "year"}, inplace=True)
    df = df[df["year"] >= starting_year]

    # Convert the columns M, F, N, M_1, M_2, etc to single column "type"
    df = df.melt(id_vars=["year"], var_name="type", value_name="true_rate")
    df["type"] = df.apply(
        lambda x: x["type"].replace("+", ""), axis=1
    )

    # Remove NA values from the true_rate column
    df = df.dropna(subset=["true_rate"])

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
    df = df.loc[df["age"] >= 3]

    # Sort by year, sex, and age
    df = df.sort_values(by=["year", "sex", "age"])

    return df


def exacerbation_calibrator(
    province: str = "CA", starting_year: int = 2000, max_cal_year: int = 2065,
    stablization_year: int = 2025, min_age: int = MIN_AGE, max_age: int = MAX_AGE,
    prob_hosp: float = 0.026, projection_scenario: str = "M3"
):
    if province == "CA":
        max_cal_year = 2065
    else:
        max_cal_year = 2043

    df_prev_inc = pd.read_csv(get_data_path("processed_data/master_asthma_prev_inc.csv"))
    df_prev = df_prev_inc[["year", "age", "sex", "prev"]]

    # Canada Institute for Health Information (CIHI) data on hospitalizations due to asthma
    df_cihi = load_hospitalization_data(province, starting_year)

    final_year = max(df_cihi["year"])
    future_years = list(range(final_year + 1, max_cal_year + 1))
    
    # Append a copy of the final year data for each of the future years, changing the year column
    for year in future_years:
        df_cihi_year = df_cihi[df_cihi["year"] == final_year].copy()
        df_cihi_year["year"] = [year] * df_cihi_year.shape[0]
        df_cihi = pd.concat([df_cihi, df_cihi_year])

    # Load population data
    df_population = pd.read_csv(
        get_data_path("processed_data/birth/initial_pop_distribution_prop.csv")
    )

    # Filter the population data by province and starting year
    df_population = df_population.loc[
        (df_population["province"] == province) & (df_population["year"] >= starting_year)
    ]

    # Add a sex column and compute n_age for each sex (n_age_sex)
    df_population_male = df_population.copy()
    df_population_male["sex"] = ["M"] * df_population.shape[0]
    df_population_male["n_age_sex"] = df_population.apply(
        lambda x: x["n_age"] * x["prop_male"], axis=1
    )
    df_population_female = df_population.copy()
    df_population_female["sex"] = ["F"] * df_population.shape[0]
    df_population_female["n_age_sex"] = df_population.apply(
        lambda x: x["n_age"] * (1 - x["prop_male"]), axis=1
    )
    df_population = pd.concat([df_population_female, df_population_male])
    df_population["n_age_sex"] = df_population["n_age_sex"].astype(int)

    # Filter the population data by projection scenario and year
    df_population = df_population.loc[
        (df_population["projection_scenario"] == "past") & (df_population["year"] <= 2021)
        | (
            (df_population["projection_scenario"] == projection_scenario)
            & (df_population["year"] > 2021)
        )
    ]

    # Remove unnecessary columns
    df_population.drop(
        columns=["projection_scenario", "n_age", "prop_male", "prop", "n_birth"],
        inplace=True
    )
    df_population.rename(columns={"n_age_sex": "n"}, inplace=True)


    df_population = df_population.loc[df_population["year"] <= max_cal_year]
    df_population = df_population.loc[df_population["age"] >= min_age]

    # Set any age > max_age to the max_age
    df_population["age"] = df_population["age"].apply(
        lambda x: min(x, max_age)
    )

    # Sum the max_age rows to a single value of n
    grouped_df = df_population.groupby(["year", "sex", "age"])
    df_population["n"] = grouped_df["n"].transform(lambda x: sum(x))
    df_population.drop_duplicates(inplace=True)

    df_target = pd.merge(df_population, df_cihi, on=["year", "sex", "age"], how="left")
    df_target["true_n"] = df_target.apply(
        lambda x: x["true_rate"] * x["n"] / 100000, axis=1
    )
    df_target = pd.merge(df_target, df_prev, on=["year", "sex", "age"], how="left")
    df_target["n_asthma"] = df_target.apply(
        lambda x: x["prev"] * x["n"], axis=1
    )

    df_target["mean_annual_exacerbation"] = df_target.apply(
        lambda x: exacerbation_prediction(x["sex"], x["age"]), axis=1
    )
    df_target["expected_exacerbations"] = df_target.apply(
        lambda x: x["mean_annual_exacerbation"] * x["n_asthma"], axis=1
    )
    df_target["expected_n"] = df_target.apply(
        lambda x: prob_hosp * x["expected_exacerbations"], axis=1
    )
    df_target["calibrator_multiplier"] = df_target.apply(
        lambda x: x["true_n"] / x["expected_n"], axis=1
    )

    df = df_target[["year", "sex", "age", "calibrator_multiplier"]]
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
        df_province = exacerbation_calibrator(province)
        df = pd.concat([df, df_province], axis=0)

    df.to_csv(get_data_path("processed_data/master_calibrated_exac.csv"), index=False)


if __name__ == "__main__":
    generate_exacerbation_calibration()
