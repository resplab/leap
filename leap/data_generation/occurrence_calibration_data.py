import pathlib
import pandas as pd
import numpy as np
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import itertools
from leap.utils import get_data_path
from leap.logger import get_logger
from typing import Tuple

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

PROVINCE = "CA"
MAX_YEAR = 2065 # 2065 for CA; 2043 for BC
MIN_YEAR = 2000
STABILIZATION_YEAR = 2025
BASELINE_YEAR = 2001
MAX_AGE = 63
MAX_AGE = 9
MAX_ASTHMA_AGE = 62
MIN_ASTHMA_AGE = 3
# odds ratio between asthma prevalence at age 3 and family history (CHILD Study)
OR_ASTHMA_AGE_3 = 1.13
# odds ratio between asthma prevalence at age 5 and family history (CHILD Study)
OR_ASTHMA_AGE_5 = 2.4
# beta parameter for the antibiotic dose term in the odds ratio equation for antibiotic courses
BETA_ABX_DOSE = 0.053
# beta parameter for the age term in the odds ratio equation for antibiotic courses
BETA_ABX_AGE = -0.225
# beta parameter for the constant term in the odds ratio equation for antibiotic courses
BETA_ABX_0 = 1.711 + 0.115
INC_BETA_PARAMS = [(np.log(OR_ASTHMA_AGE_5) - np.log(OR_ASTHMA_AGE_3)) / 2, BETA_ABX_AGE]
# the probability that one or more parents have asthma (CHILD Study)
PROB_FAM_HIST = 0.2927242

DF_OCC_PRED = pd.read_csv(get_data_path("processed_data/asthma_occurrence_predictions.csv"))


def asthma_predictor(age: int, sex: str, year: int, occurrence_type: str) -> float:
    """
    Predicts the asthma prevalence or incidence based on the given parameters.
    Args:
        age: Age of the individual.
        sex: One of "M" or "F".
        year: Year of the prediction.
        occurrence_type: One of "prevalence" or "incidence".
    Returns:
        A float representing the predicted asthma prevalence or incidence.
    """

    age = min(age, MAX_ASTHMA_AGE)
    year = min(year, STABILIZATION_YEAR)

    return DF_OCC_PRED.loc[
        (DF_OCC_PRED["age"] == age) &
        (DF_OCC_PRED["year"] == year) &
        (DF_OCC_PRED["sex"] == sex)
    ][occurrence_type].values[0]


def load_occurrence_data(
    province: str = PROVINCE,
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the asthma incidence and prevalence data for the given province and year range.

    Args:
        province: The province to load data for.
        min_year: The minimum year to load data for.
        max_year: The maximum year to load data for.

    Returns:
        A tuple of two DataFrames: the first is the incidence data, and the second is the
        prevalence data.
    """

    df_asthma = pd.DataFrame(
        list(itertools.product(range(3, 111), [0, 1], range(min_year, max_year + 1))),
        columns=["age", "sex", "year"]
    )

    df_asthma["incidence"] = df_asthma.apply(
        lambda x: asthma_predictor(x["age"], x["sex"], x["year"], "inc"),
        axis=1
    )
    df_asthma["prevalence"] = df_asthma.apply(
        lambda x: asthma_predictor(x["age"], x["sex"], x["year"], "prev"),
        axis=1
    )
    df_asthma["incidence"] = df_asthma.apply(
        lambda x: x["prevalence"] if x["age"] == 3 else x["incidence"],
        axis=1
    )
    df_asthma["province"] = [province] * df_asthma.shape[0]

    df_incidence = df_asthma[["year", "age", "sex", "incidence"]].copy()
    df_prevalence = df_asthma[["year", "age", "sex", "prevalence"]].copy()

    return df_incidence, df_prevalence


def load_reassessment_data(
    province: str = PROVINCE
) -> pd.DataFrame:
    """Load the asthma reassessment data for the given province.

    Args:
        province: The province to load data for.

    Returns:
        A DataFrame containing the reassessment data.
    """

    df_reassessment = pd.read_csv(get_data_path("processed_data/asthma_reassessment.csv"))
    df_reassessment = df_reassessment.loc[df_reassessment["province"] == province]

    df_reassessment = df_reassessment.drop(columns=["province"])

    return df_reassessment


def load_family_history_data() -> pd.DataFrame:
    """Load the family history data for the given province.

    Returns:
        A dataframe containing the family history odds ratios.
        It contains the following columns:

            * ``age (int)``: The age of the individual. Ranges from ``3`` to ``5``.
            * ``fam_history (int)``: Whether or not there is a family history of asthma.
              ``0`` = one or more parents has asthma, 
              ``1`` = neither parent has asthma.
            * ``odds_ratio (float)``: The odds ratio for asthma prevalence based on family
                history and age. The odds ratio is calculated based on the CHILD study data.
    """

    df_fam_history_or = pd.DataFrame(
        list(itertools.product(range(3, 6), [0, 1])),
        columns=["age", "fam_history"]
    )
    df_fam_history_or["odds_ratio"] = [
        1, OR_ASTHMA_AGE_3,
        1, np.exp((np.log(OR_ASTHMA_AGE_3) + np.log(OR_ASTHMA_AGE_5)) / 2),
        1, OR_ASTHMA_AGE_5
    ]

    return df_fam_history_or


def load_abx_exposure_data() -> pd.DataFrame:
    """Load the antibiotic exposure data.

    Returns:
        A dataframe containing the antibiotic exposure odds ratios.
        It contains the following columns:

            * ``age (int)``: The age of the individual. An integer in ``[3, 8]``.
            * ``abx_dose (int)``: The number of antibiotic courses taken in the first year of life,
              an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
            * ``odds_ratio (float)``: The odds ratio for asthma prevalence based on antibiotic
              exposure during the first year of life and age. The odds ratio is calculated based on
              the CHILD study data.
    """

    df_abx_or = pd.read_csv(get_data_path("original_data/dose_response_log_aOR.csv"))
    df_abx_or.columns = ["age"] + [i for i in range(1, 6)]
    df_abx_or.insert(1, 0, [0] * df_abx_or.shape[0])

    for abx_dose in range(0, 6):
        df_abx_or[abx_dose] = df_abx_or[abx_dose].apply(
            lambda x: np.exp(x)
        )

    df_abx_or = df_abx_or.loc[df_abx_or["age"] >= 3]
    df_abx_or = pd.concat(
        [df_abx_or, pd.DataFrame({"age": [8], 0: [1], 1: [1], 2: [1], 3: [1], 4: [1], 5: [1]})],
        ignore_index=True
    )

    df_abx_or = df_abx_or.melt(id_vars=["age"], var_name="abx_dose", value_name="odds_ratio")
    return df_abx_or


def OR_abx_calculator(
    age: int,
    dose: int,
    params: list[float] | None = None
) -> float:
    """Calculate the odds ratio for asthma prevalence based on antibiotic exposure.

    Args:
        age: The age of the individual in years.
        dose: The number of antibiotic courses taken in the first year of life,
            an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
        params: The parameters for the odds ratio calculation. If ``None``, the default
            parameters are used. The default parameters are:

            * ``BETA_ABX_0``: The beta parameter for the constant term in the odds
                ratio equation for antibiotic courses.
            * ``BETA_ABX_AGE``: The beta parameter for the age term in the odds
                ratio equation for antibiotic courses.
            * ``BETA_ABX_DOSE``: The beta parameter for the dose term in the odds
                ratio equation for antibiotic courses.

    Returns:
        A float representing the odds ratio for asthma prevalence based on antibiotic
        exposure and age.
    """
    if params is None:
        params = [BETA_ABX_0, BETA_ABX_AGE, BETA_ABX_DOSE]

    if dose == 0:
        return 1.0
    else:
        return np.exp(np.sum(np.dot(params, [1, min(age, 7), min(dose, 3)])))


def OR_fam_calculator(
    age: int,
    fam_hist: int,
    params: list[float] | None
) -> float:
    """Calculate the odds ratio for asthma prevalence based on family history.

    Args:
        age: The age of the individual in years.
        fam_hist: Whether or not there is a family history of asthma.
            ``0`` = one or more parents has asthma,
            ``1`` = neither parent has asthma.
        params: The parameters for the odds ratio calculation. If ``None``, the default
            parameters are used.
    
    Returns:
        A float representing the odds ratio for asthma prevalence based on family
        history and age.
    """
    if params is None:
        params = [np.log(OR_ASTHMA_AGE_3), (np.log(OR_ASTHMA_AGE_5) + np.log(OR_ASTHMA_AGE_3)) / 2]

    if age < MIN_ASTHMA_AGE or fam_hist == 0 or age > 7:
        return 1.0
    else:
        return np.exp(params[0] + params[1] * (min(age, 5) - 3))


def OR_risk_factor_calculator(
    fam_hist: int,
    age: int,
    dose: int,
    family_history_params: list[float] | None = None,
    abx_params: list[float] | None = None
) -> float:
    """Calculate the odds ratio for asthma prevalence based on family history and antibiotic exposure.

    Args:
        fam_hist: Whether or not there is a family history of asthma.
            ``0`` = one or more parents has asthma,
            ``1`` = neither parent has asthma.
        age: The age of the individual in years.
        dose: The number of antibiotic courses taken in the first year of life,
            an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
        family_history_params: The parameters for the family history odds ratio calculation.
            If ``None``, the default parameters are used.
        abx_params: The parameters for the antibiotic exposure odds ratio calculation.
            If ``None``, the default parameters are used.

    Returns:
        The odds ratio for asthma prevalence based on family history and antibiotic exposure.
    """
    if family_history_params is None:
        family_history_params = [
            np.log(OR_ASTHMA_AGE_3),
            (np.log(OR_ASTHMA_AGE_3) + np.log(OR_ASTHMA_AGE_5)) / 2 - np.log(OR_ASTHMA_AGE_3)
        ]

    if abx_params is None:
        abx_params = [BETA_ABX_0, BETA_ABX_AGE, BETA_ABX_DOSE]

    if age < MIN_ASTHMA_AGE:
        return 1
    else:
        return np.exp(
            np.log(OR_fam_calculator(age, fam_hist, family_history_params)) +
            np.log(OR_abx_calculator(age, dose, abx_params))
        )
    

def risk_factor_generator(
    year: int,
    age: int,
    sex: str,
    model_abx: GLMResultsWrapper,
    p_fam_distribution: pd.DataFrame,
    df_fam_history_or: pd.DataFrame,
    df_abx_or: pd.DataFrame
) -> pd.DataFrame:
    """Compute the combined antibiotic exposure and family history odds ratio.

    Args:
        year: The current year.
        age: The age of the person in years.
        sex: One of ``M`` or ``F``.
        model_abx: The fitted ``Negative Binomial`` model for the number of courses of antibiotics.
        p_fam_distribution: A dataframe with the probability of family history of asthma, given
            that the person has asthma. Columns:
            * ``fam_history (int)``: Whether or not there is a family history of asthma.
              ``0`` = one or more parents has asthma
              ``1`` = neither parent has asthma
            * ``prob_fam``: The probability of family history of asthma, given that the person
              has asthma.
        df_fam_history_or: A dataframe with the odds ratio of family history of asthma, given
            the age of the person. Columns:
            * ``age (int)``: An integer in ``[3, 5]``.
            * ``fam_history (int)``: Whether or not there is a family history of asthma.
              ``0`` = one or more parents has asthma
              ``1`` = neither parent has asthma
            * ``odds_ratio (float)``: The odds ratio of family history of asthma, given that the
              person has asthma. The odds ratio is calculated based on the CHILD study data.
        df_abx_or: A dataframe with the odds ratio of antibiotic exposure, given
            the age of the person. Columns:
            * ``age (int)``: An integer in ``[3, 8]``.
            * ``abx_exposure (int)``: The number of antibiotic courses taken in the first year of
              life; an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
            * ``odds_ratio (float)``: The odds ratio of antibiotic exposure, given that the person
              has asthma.

    Returns:
        A dataframe with the combined probabilities and odds ratios for the antibiotic exposure
        and family history risk factors.
        Columns:

            * ``fam_history (int)``: Whether or not there is a family history of asthma.
              ``0`` = one or more parents has asthma
              ``1`` = neither parent has asthma
            * ``abx_exposure (int)``: The number of antibiotic courses taken in the first year of
              life; an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
            * ``year (int)``: The given year.
            * ``sex (str)``: One of ``M`` or ``F``.
            * ``age (int)``: The age of the person in years.
            * ``prob (float)``: The probability of antibiotic exposure * probability of one or
              more parents having asthma given that the person has asthma.
            * ``odds_ratio (float)``: The odds combined odds ratio:
              ``odds_ratio = odds_ratio_abx * odds_ratio_fam_hist``
    """


    birth_year = year - age
    df_abx_exposure = p_antibiotic_exposure(max(birth_year, 2000), sex, model_abx)

    # combine abx_exposure = 3, 4, 5+ into 3+
    df_abx_exposure.loc[df_abx_exposure["abx_exposure"] == 3, "prob_abx"] = df_abx_exposure.loc[
        df_abx_exposure["abx_exposure"] >= 3, "prob_abx"
    ].sum()
    df_abx_exposure = df_abx_exposure[df_abx_exposure["abx_exposure"] <= 3]

    # select the given age if <= 5, otherwise select age == 5
    df_fam_history_or_age = df_fam_history_or.loc[df_fam_history_or["age"] == min(age, 5)]
    df_fam_history_or_age = df_fam_history_or_age[["fam_history", "odds_ratio"]]
    df_fam_history_or_age.rename(columns={"odds_ratio": "odds_ratio_fam_hist"}, inplace=True)
  
    # select the given age if <= 8, otherwise select age == 8
    # filter out abx_exposure > 3
    df_abx_or_age = df_abx_or.loc[df_abx_or["age"] == min(age, 8)]
    df_abx_or_age = df_abx_or_age[["abx_exposure", "odds_ratio"]]
    df_abx_or_age.rename(columns={"odds_ratio": "odds_ratio_abx"}, inplace=True)
    df_abx_or_age = df_abx_or_age.loc[df_abx_or_age["abx_exposure"] <= 3]
       
    df_risk_factors = pd.DataFrame(
        list(itertools.product([0, 1], range(0, 4))),
        columns=["fam_history", "abx_exposure"]
    )
    df_risk_factors["year"] = [year] * df_risk_factors.shape[0]
    df_risk_factors["sex"] = [sex] * df_risk_factors.shape[0]
    df_risk_factors["age"] = [age] * df_risk_factors.shape[0]

    df_risk_factors = pd.merge(
        df_risk_factors,
        p_fam_distribution,
        how="left",
        on="fam_history"
    )
    df_risk_factors = pd.merge(
        df_risk_factors,
        df_abx_exposure,
        how="left",
        on="abx_exposure"
    )
    df_risk_factors = pd.merge(
        df_risk_factors,
        df_fam_history_or_age,
        how="left",
        on="fam_history"
    )
    df_risk_factors = pd.merge(
        df_risk_factors,
        df_abx_or_age,
        how="left",
        on="abx_exposure"
    )
    df_risk_factors["prob"] = df_risk_factors.apply(
        lambda x: x["prob_fam"] * x["prob_abx"],
        axis=1
    )
    df_risk_factors["odds_ratio"] = df_risk_factors.apply(
        lambda x: x["odds_ratio_fam_hist"] * x["odds_ratio_abx"],
        axis=1
    )
    df_risk_factors = df_risk_factors[
        ["fam_history", "abx_exposure", "year", "sex", "age", "prob", "odds_ratio"]
    ]
    return df_risk_factors



