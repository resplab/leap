import pandas as pd
import numpy as np
import re
import json
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
    "θ": [-0.3950, 2.754]
}


# Annual average exacerbation rate for a person with asthma
# Source: Economic Burden of Asthma (EBA) study (Chen et al., 2013)
# https://bmjopen.bmj.com/content/3/9/e003360.short
EXACERBATION_RATE = 0.347

# The ratio between the exacerbation rates for different control levels
# rate(exacerbation | fully controlled) = 1
# rate(exacerbation | partially controlled) = 2 * rate(exacerbation | fully controlled)
# rate(exacerbation | uncontrolled) = 3 * rate(exacerbation | fully controlled)
# Source: GOAL Study (Bateman et al., 2004)
# https://www.atsjournals.org/doi/full/10.1164/rccm.200401-033OC
EXACERBATION_RATE_RATIO = {
    "well_controlled": 1.0,
    "partially_controlled": 2.0,
    "uncontrolled": 3.0
}

# The proportion of time spent in each control level
# Source: Economic Burden of Asthma (EBA) study (Chen et al., 2013)
# https://bmjopen.bmj.com/content/3/9/e003360.short
CONTROL_LEVEL_PROPORTIONS = {
    "well_controlled": 0.340,
    "partially_controlled": 0.474,
    "uncontrolled": 0.186
}


def compute_beta_control(
    rate: float = EXACERBATION_RATE,
    rate_ratio: dict[str, float] = EXACERBATION_RATE_RATIO,
    control_level_proportions: dict[str, float] = CONTROL_LEVEL_PROPORTIONS
) -> np.ndarray:
    """Compute the beta control parameters for the exacerbation model.

    Args:
        rate: The annual exacerbation rate for a person with asthma.

    Returns:
        A list of three floats, the beta control parameters.
    """
    denominator = 0
    for level in rate_ratio:
        denominator += control_level_proportions[level] * rate_ratio[level]
    rate_wc = rate / denominator
    rate_pc = rate_wc * rate_ratio["partially_controlled"]
    rate_uc = rate_wc * rate_ratio["uncontrolled"]
    return np.log([rate_wc, rate_pc, rate_uc])

    


def exacerbation_prediction(
    sex: str, age: int, beta_control: np.ndarray
):
    r"""Calculate the mean number of exacerbations for a given age and sex.

    .. math::

        \ln(\lambda_{C}) = \sum_{i=1}^3 \beta_i c_i

    where:

    * :math:`\lambda_{C}` is the predicted average number of asthma exacerbations per year.
    * :math:`\beta_i` is the control parameter.
    * :math:`c_i` is the relative time spent in control level :math:`i`.

    Here the :math:`\beta_i` values were calculated from the
    `Economic Burden of Asthma (EBA) study <https://bmjopen.bmj.com/content/3/9/e003360.long>`_
    and the `GOAL Study <https://www.atsjournals.org/doi/full/10.1164/rccm.200401-033OC>`_.

    Args:
        sex: One of "M" or "F".
        age: Integer age, a value in ``[3, 90]``.
        beta_control: A list of three floats, the control parameters.

    Returns:
        The predicted number of exacerbations per year per person with asthma.
    """
    if age < 3:
        return 0
    else:
        control = Control(parameters=CONTROL_PARAMETERS, hyperparameters=None)
        control_levels = control.compute_control_levels(sex=Sex(sex), age=age)
        return np.exp(np.sum(control_levels.as_array() * beta_control))


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
    `Canadian Institute for Health Information (CIHI) 
    <https://www.cihi.ca/en/hospital-morbidity-database-hmdb-metadata>`_.

    The hospitalization data was collected from patients presenting to a hospital in Canada
    due to an asthma exacerbation. We will use this data to calibrate the exacerbation model.
    
    Args:
        province: The province for which to load the hospitalization data.
        starting_year: The starting year for which to load the hospitalization data.
        min_age: The minimum age for to be used in the data. We are assuming that
            asthma diagnoses are made at age 3 and older, so the default is 3.
        
    Returns:
        The hospitalization data for the given province and starting year.
        Columns:

        * ``year``: The year of the data.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``age``: Integer age, a value in ``[3, 90]``.
        * ``hospitalization_rate``: The observed number of hospitalizations per ``100 000`` people
          for a given year, age, and sex.
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
        A dataframe containing the Canadian population data.
        Columns:

        * ``year``: The year of the data.
        * ``age``: A value in ``[min_age, max_age]``.
        * ``province``: The 2-letter province abbreviation.
        * ``sex``: One of ``M`` = male, ``F`` = female.
        * ``n``: The number of people in a given year, age, sex, province, and projection scenario.

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
    beta_control: np.ndarray,
    province: str = "CA",
    starting_year: int = 2000,
    max_year: int = 2065,
    min_age: int = MIN_AGE,
    max_age: int = MAX_AGE,
    prob_hosp: float = PROB_HOSP,
    projection_scenario: str = "M3"
) -> pd.DataFrame:
    """Compute the ratio between the observed and predicted hospitalization rates.
    
    Args:
        beta_control: A list of three floats, the control beta coefficients.
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

    Returns:
        A dataframe with the following columns:

        * ``year``: The year of the data, a value in ``[starting_year, max_year]``.
        * ``age``: The integer age, a value in ``[min_age, max_age]``.
        * ``sex``: One of ``M`` or ``F``.
        * ``calibrator_multiplier``: The ratio between the observed and predicted number of
          hospitalizations.
    """

    df_prev_inc = pd.read_csv(get_data_path("processed_data/asthma_occurrence_predictions.csv"))
    df_prev = df_prev_inc[["year", "age", "sex", "prevalence"]]
    df_prev["sex"] = df_prev.apply(
        lambda x: "F" if x["sex"]==0 else "M", axis=1
    )

    # Canada Institute for Health Information (CIHI) data on hospitalizations due to asthma
    df_hosp = load_hospitalization_data(province, starting_year, min_age)

    final_year = max(df_hosp["year"])
    future_years = list(range(final_year + 1, max_year + 1))
    
    # Append a copy of the final year data for each of the future years, changing the year column
    for year in future_years:
        df_hosp_year = df_hosp[df_hosp["year"] == final_year].copy()
        df_hosp_year["year"] = [year] * df_hosp_year.shape[0]
        df_hosp = pd.concat([df_hosp, df_hosp_year])

    # Load population data
    df_population = load_population_data(
        province, starting_year, projection_scenario, max_year, min_age, max_age
    )

    # Calculate the number of hospitalizations for a given year, age, and sex
    # hospitalization_rate: The observed number of hospitalizations per 100 000 people.
    # n: The number of people in a given year, age, and sex.
    df_target = pd.merge(df_population, df_hosp, on=["year", "sex", "age"], how="left")
    df_target["n_hosp"] = df_target.apply(
        lambda x: x["hospitalization_rate"] * x["n"] / 100000, axis=1
    )

    # Calculate the number of people with asthma for a given year, age, and sex
    # prev: The prevalence of asthma for a given year, age, and sex.
    # n: The number of people in a given year, age, and sex.
    df_target = pd.merge(df_target, df_prev, on=["year", "sex", "age"], how="left")
    df_target["n_asthma"] = df_target.apply(
        lambda x: x["prevalence"] * x["n"], axis=1
    )

    # Calculate the mean number of exacerbations for a given age and sex
    df_target["mean_annual_exacerbation"] = df_target.apply(
        lambda x: exacerbation_prediction(x["sex"], x["age"], beta_control), axis=1
    )

    # Calculate the predicted number of exacerbations for a given year, age, and sex
    # mean_annual_exacerbation: The mean number of exacerbations for a given year, age, and sex.
    # n_asthma: The number of people with asthma for a given age, sex, and year.
    df_target["n_exacerbations_pred"] = df_target.apply(
        lambda x: x["mean_annual_exacerbation"] * x["n_asthma"], axis=1
    )

    # Calculate the predicted number of hospitalizations for a given year, age, and sex.
    # prob_hosp: the number of exacerbations per year per person with asthma.
    # n_exacerbations_pred: The predicted number of exacerbations for a given year, age, and sex.
    df_target["n_hosp_pred"] = df_target.apply(
        lambda x: prob_hosp * x["n_exacerbations_pred"], axis=1
    )

    # Calculate the ratio between the observed and predicted number of hospitalizations
    df_target["calibrator_multiplier"] = df_target.apply(
        lambda x: x["n_hosp"] / x["n_hosp_pred"], axis=1
    )

    # Drop unnecessary columns
    df = df_target[["year", "sex", "age", "calibrator_multiplier"]]

    # Add province column
    df["province"] = [province] * df.shape[0]

    return df



def generate_exacerbation_calibration_data():
    """Generate the exacerbation calibration data for all provinces."""

    beta_control = compute_beta_control()

    df = pd.DataFrame({
        "year": np.array([], dtype=int),
        "sex": [],
        "age": np.array([], dtype=int),
        "calibrator_multiplier": []
    })
    for province in PROVINCES:
        df_province = exacerbation_calibrator(beta_control, province, max_year=MAX_YEARS[province])
        df = pd.concat([df, df_province], axis=0)


    # Update the config file with the beta coefficients
    config_path = get_data_path("processed_data/config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["exacerbation"]["parameters"] = {
        "βcontrol_C": beta_control[0],
        "βcontrol_PC": beta_control[1],
        "βcontrol_UC": beta_control[2]
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logger.message("Exacerbation beta coefficients generated and saved to config.json")

    # Save the calibration data
    df.to_csv(get_data_path("processed_data/exacerbation_calibration.csv"), index=False)


if __name__ == "__main__":
    generate_exacerbation_calibration_data()
