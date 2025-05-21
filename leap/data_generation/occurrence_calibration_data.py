import pandas as pd
import json
import numpy as np
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from scipy import optimize
import itertools
from leap.utils import get_data_path
from leap.data_generation.prevalence_calibration import optimize_prevalence_β_parameters, \
    compute_asthma_prevalence_λ, get_asthma_prevalence_correction
from leap.data_generation.incidence_calibration import inc_correction_calculator, \
    compute_odds_ratio_difference
from leap.data_generation.antibiotic_data import get_predicted_abx_data, generate_antibiotic_data
from leap.logger import get_logger
from typing import Tuple, Dict
from scipy.stats import nbinom
from scipy.special import logit

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
MAX_ABX_AGE = 7
# odds ratio between asthma prevalence at age 3 and family history (CHILD Study)
OR_ASTHMA_AGE_3 = 1.13
# odds ratio between asthma prevalence at age 5 and family history (CHILD Study)
OR_ASTHMA_AGE_5 = 2.4
# beta parameter for the constant term in the odds ratio equation for family history
BETA_FHX_0 = np.log(OR_ASTHMA_AGE_3)
# beta parameter for the age term in the odds ratio equation for family history
BETA_FHX_AGE = (np.log(OR_ASTHMA_AGE_5) - np.log(OR_ASTHMA_AGE_3)) / 2
# beta parameter for the antibiotic dose term in the odds ratio equation for antibiotic courses
BETA_ABX_DOSE = 0.053
# beta parameter for the age term in the odds ratio equation for antibiotic courses
BETA_ABX_AGE = -0.225
# beta parameter for the constant term in the odds ratio equation for antibiotic courses
# default beta parameters for the risk factors equations
β_RISK_FACTORS = {
    "fam_history": {
        "β_fhx_0": BETA_FHX_0,
        "β_fhx_age": BETA_FHX_AGE
    },
    "abx": {
        "β_abx_0": BETA_ABX_0, 
        "β_abx_age": BETA_ABX_AGE,
        "β_abx_dose": BETA_ABX_DOSE
    }
}
# the probability that one or more parents have asthma (CHILD Study)
PROB_FAM_HIST = 0.2927242

DF_OCC_PRED = pd.read_csv(get_data_path("processed_data/asthma_occurrence_predictions.csv"))


def asthma_predictor(age: int, sex: str, year: int, occurrence_type: str) -> float:
    """Predicts the asthma prevalence or incidence based on the given parameters.

    Args:
        age: Age of the individual in years.
        sex: One of ``"M"`` or ``"F"``.
        year: Year of the prediction.
        occurrence_type: One of ``"prevalence"`` or ``"incidence"``.

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
        list(itertools.product(range(3, 111), ["F", "M"], range(min_year, max_year + 1))),
        columns=["age", "sex", "year"]
    )

    df_asthma["incidence"] = df_asthma.apply(
        lambda x: asthma_predictor(x["age"], x["sex"], x["year"], "incidence"),
        axis=1
    )
    df_asthma["prevalence"] = df_asthma.apply(
        lambda x: asthma_predictor(x["age"], x["sex"], x["year"], "prevalence"),
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


def load_family_history_data(β_fam_history: Dict[str, float] | None = None) -> pd.DataFrame:
    """Load the family history data for the given province.

    Args:
        β_fam_history: A dictionary of two beta parameters for the odds ratio calculation:
            * ``β_fhx_0``: The beta parameter for the constant term in the equation
            * ``β_fhx_age``: The beta parameter for the age term in the equation.

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

    df_fam_history_or["odds_ratio"] = df_fam_history_or.apply(
        lambda x: calculate_odds_ratio_fam_history(x["age"], x["fam_history"], β_fam_history),
        axis=1
    )

    return df_fam_history_or


def load_abx_exposure_data(β_abx: Dict[str, float] | None = None) -> pd.DataFrame:
    """Load the antibiotic exposure data.

    Args:
        β_abx: A dictionary of 3 beta parameters for the odds ratio calculation:
            * ``β_abx_0``: The beta parameter for the constant term in the equation
            * ``β_abx_age``: The beta parameter for the age term in the equation
            * ``β_abx_dose``: The beta parameter for the antibiotic dose term in the equation.

    Returns:
        A dataframe with the odds ratios of asthma prevalence given the number of courses of
        antibiotics taken during the first year of life.
        It contains the following columns:

            * ``age (int)``: The age of the individual. An integer in ``[3, 8]``.
            * ``abx_dose (int)``: The number of antibiotic courses taken in the first year of life,
              an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
            * ``odds_ratio (float)``: The odds ratio for asthma prevalence based on antibiotic
              exposure during the first year of life and age. The odds ratio is calculated based on
              the CHILD study data.
    """

    df_abx_or = pd.DataFrame(
        list(itertools.product(range(3, 9), range(0, 4))),
        columns=["age", "n_abx"]
    )

    df_abx_or["odds_ratio"] = df_abx_or.apply(
        lambda x: calculate_odds_ratio_abx(x["age"], x["n_abx"], β_abx),
        axis=1
    )
    return df_abx_or


def p_antibiotic_exposure(
    year: int,
    sex: str,
    model_abx: GLMResultsWrapper
) -> pd.DataFrame:
    """Compute the probability of number of courses of antibiotics during infancy.

    Args:
        year: The birth year of the person.
        sex: The sex of the infant, one of ``"M"`` or ``"F"``.
        model_abx: The fitted ``Negative Binomial`` model for the number of courses of antibiotics.
            This model was fitted using BC Ministry of Health data on antibiotic prescriptions.
    
    Returns:
        A dataframe with the probability of the number of courses of antibiotics,
        ranging from 0 - 5+.
        Columns:

            * ``n_abx (int)``: The number of courses of antibiotics, an integer in ``[0, 5]``,
              where ``5`` indicates 5 or more courses.
            * ``prob (float)``: The probability that a person of the given sex and birth year
              was given ``n_abx`` courses of antibiotics during the first year of their life.
    """

    if sex == "M":
        year = min(2028 - 1, year)
    else:
        year = min(2025 - 1, year)

    df = pd.DataFrame({
        "year": [year], "sex": [sex], "n_abx": [1]
    })

    # Get the predicted average number of courses of antibiotics
    df = get_predicted_abx_data(
        model=model_abx,
        df=df
    )

    # μ: average number of courses of antibiotics
    # θ: dispersion parameter
    μ = df["n_abx_μ"].iloc[0]
    θ = 1 / model_abx.family.alpha
    p = θ / (θ + μ)

    # Calculate probability of 0, 1, 2, 3, 4, 5+ courses of antibiotics via negative binomial
    prob = nbinom.pmf(list(range(0, 5)), n=θ, p=p)
    prob = np.append(prob, 1 - prob.sum())
    df = pd.DataFrame({
        "n_abx": list(range(0, 6)),
        "prob": prob
    })
    return df




def calculate_odds_ratio_abx(
    age: int,
    dose: int,
    β_abx: Dict[str, float] | None = None
) -> float:
    """Calculate the odds ratio for asthma prevalence based on antibiotic exposure.

    Args:
        age: The age of the individual in years.
        dose: The number of antibiotic courses taken in the first year of life,
            an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
        β_abx: The parameters for the odds ratio calculation. If ``None``, the default
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
    if β_abx is None:
        β_abx = β_RISK_FACTORS["abx"]

    if dose == 0:
        return 1.0
    else:
        return np.exp(
            β_abx["β_abx_0"] + 
            β_abx["β_abx_age"] * min(age, MAX_ABX_AGE) + 
            β_abx["β_abx_dose"] * min(dose, 3)
        )


def calculate_odds_ratio_fam_history(
    age: int,
    fam_hist: int,
    β_fam_hist: Dict[str, float] | None = None
) -> float:
    """Calculate the odds ratio for asthma prevalence based on family history.

    Args:
        age: The age of the individual in years.
        fam_hist: Whether or not there is a family history of asthma.
            ``0`` = one or more parents has asthma,
            ``1`` = neither parent has asthma.
        β_fam_hist: The beta parameters for the odds ratio calculation:
            * ``β_fhx_0``: The beta parameter for the constant term in the odds ratio equation
            * ``β_fhx_age``: The beta parameter for the age term in the odds ratio equation.
    
    Returns:
        A float representing the odds ratio for asthma prevalence based on family
        history and age.
    """
    if β_fam_hist is None:
        β_fam_hist = β_RISK_FACTORS["fam_history"]

    if age < MIN_ASTHMA_AGE or fam_hist == 0 or age > MAX_ABX_AGE:
        return 1.0
    else:
        return np.exp(
            β_fam_hist["β_fhx_0"] + β_fam_hist["β_fhx_age"] * (min(age, 5) - MIN_ASTHMA_AGE)
        )


def calculate_odds_ratio_risk_factors(
    fam_hist: int,
    age: int,
    dose: int,
    β_risk_factors: Dict[str, Dict[str, float]] | None = None
) -> float:
    """Calculate the odds ratio for asthma prevalence based on family history and antibiotic exposure.

    Args:
        fam_hist: Whether or not there is a family history of asthma.
            ``0`` = one or more parents has asthma,
            ``1`` = neither parent has asthma.
        age: The age of the individual in years.
        dose: The number of antibiotic courses taken in the first year of life,
            an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
        β_risk_factors: A dictionary of beta parameters for the risk factor equations.
            Must contain the following keys:
            * ``fam_history``: A dictionary with the beta parameters for the family history
              odds ratio calculation. Must contain the keys ``β_fhx_0`` and ``β_fhx_age``.
            * ``abx``: A dictionary with the beta parameters for the antibiotic exposure
              odds ratio calculation. Must contain the keys ``β_abx_0``, ``β_abx_age``,
              and ``β_abx_dose``.

    Returns:
        The odds ratio for asthma prevalence based on family history and antibiotic exposure.
    """
    if β_risk_factors is None:
        β_risk_factors = β_RISK_FACTORS

    if age < MIN_ASTHMA_AGE:
        return 1.0
    else:
        odds_ratio_fam_history = calculate_odds_ratio_fam_history(
            age, fam_hist, β_risk_factors.get("fam_history", None)
        )
        odds_ratio_abx = calculate_odds_ratio_abx(age, dose, β_risk_factors.get("abx", None))
        return odds_ratio_fam_history * odds_ratio_abx
    

def risk_factor_generator(
    year: int,
    age: int,
    sex: str,
    model_abx: GLMResultsWrapper,
    β_fam_history: Dict[str, float] | None = None,
    β_abx: Dict[str, float] | None = None
) -> pd.DataFrame:
    """Compute the combined antibiotic exposure and family history odds ratio.

    Args:
        year: The current year.
        age: The age of the person in years.
        sex: One of ``M`` or ``F``.
        model_abx: The fitted ``Negative Binomial`` model for the number of courses of antibiotics.
        β_fam_history: A dictionary of 2 beta parameters for the calculation of the odds
            ratio of having asthma given family history:
            * ``β_fhx_0``: The beta parameter for the constant term in the equation.
            * ``β_fhx_age``: The beta parameter for the age term in the equation.
        β_abx: A dictionary of 3 beta parameters for the calculation of the odds
            ratio of having asthma given antibiotic exposure during infancy:
            * ``β_abx_0``: The beta parameter for the constant term in the equation.
            * ``β_abx_age``: The beta parameter for the age term in the equation.
            * ``β_abx_dose``: The beta parameter for the antibiotic dose term in the equation.

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
    df_abx_prob = p_antibiotic_exposure(max(birth_year, 2000), sex, model_abx)

    # combine abx_exposure = 3, 4, 5+ into 3+
    df_abx_prob.loc[df_abx_prob["n_abx"] == 3, "prob"] = df_abx_prob.loc[
        df_abx_prob["n_abx"] >= 3, "prob"
    ].sum()
    df_abx_prob = df_abx_prob[df_abx_prob["n_abx"] <= 3]
    df_abx_prob.rename(columns={"prob": "prob_abx"}, inplace=True)


    # select the given age if <= 5, otherwise select age == 5
    df_fam_history_or = load_family_history_data(β_fam_history)
    df_fam_history_or_age = df_fam_history_or.loc[df_fam_history_or["age"] == min(age, 5)]
    df_fam_history_or_age = df_fam_history_or_age[["fam_history", "odds_ratio"]]
    df_fam_history_or_age.rename(columns={"odds_ratio": "odds_ratio_fam_hist"}, inplace=True)
  
    # select the given age if <= 8, otherwise select age == 8
    # filter out abx_exposure > 3
    df_abx_or = load_abx_exposure_data(β_abx)
    df_abx_or_age = df_abx_or.loc[df_abx_or["age"] == min(age, MAX_ABX_AGE + 1)]
    df_abx_or_age = df_abx_or_age[["n_abx", "odds_ratio"]]
    df_abx_or_age.rename(columns={"odds_ratio": "odds_ratio_abx"}, inplace=True)
    df_abx_or_age = df_abx_or_age.loc[df_abx_or_age["n_abx"] <= 3]

    # family history probabilities (from the CHILD study)
    df_fam_history_prob = pd.DataFrame({
        "fam_history": [0, 1],
        "prob_fam": [1 - PROB_FAM_HIST, PROB_FAM_HIST]
    })
       
    df_risk_factors = pd.DataFrame(
        list(itertools.product([0, 1], range(0, 4))),
        columns=["fam_history", "n_abx"]
    )
    df_risk_factors["year"] = [year] * df_risk_factors.shape[0]
    df_risk_factors["sex"] = [sex] * df_risk_factors.shape[0]
    df_risk_factors["age"] = [age] * df_risk_factors.shape[0]

    df_risk_factors = pd.merge(
        df_risk_factors,
        df_fam_history_prob,
        how="left",
        on="fam_history"
    )
    df_risk_factors = pd.merge(
        df_risk_factors,
        df_abx_prob,
        how="left",
        on="n_abx"
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
        on="n_abx"
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
        ["fam_history", "n_abx", "year", "sex", "age", "prob", "odds_ratio"]
    ]
    return df_risk_factors


def calibrator(
    year: int,
    sex: str,
    age: int,
    model_abx: GLMResultsWrapper,
    df_incidence: pd.DataFrame,
    df_prevalence: pd.DataFrame,
    df_reassessment: pd.DataFrame,
    β_risk_factors: Dict[str, Dict[str, float]] = β_RISK_FACTORS,
    min_year: int = MIN_YEAR
) -> pd.Series:
    """Compute the loss function given the effects of risk factors in the incidence equation, for
    each year, age, and sex.

    Args:
        year: The integer year.
        age: The age in years.
        sex: One of ``M`` = male, ``F`` = female.
        model_abx: The fitted ``Negative Binomial`` model for the number of courses of antibiotics.
        df_incidence: A dataframe with the incidence of asthma, with the following columns:
            * ``year (int)``: the year
            * ``age (int)``: the age in years
            * ``sex (str)``: ``M`` = male, ``F`` = female
            * ``incidence (float)``: the incidence of asthma for the given year, age, and sex.
        df_prevalence: A dataframe with the prevalence of asthma, with the following columns:
            * ``year (int)``: the year
            * ``age (int)``: the age in years
            * ``sex (str)``: ``M`` = male, ``F`` = female
            * ``prevalence (float)``: the prevalence of asthma for the given year, age, and sex.
        df_reassessment: A dataframe with the reassessment of asthma, with the following columns:
            * ``year (int)``: the year
            * ``age (int)``: the age in years
            * ``sex (str)``: ``M`` = male, ``F`` = female
            * ``ra (float)``: the reassessment of asthma
        β_risk_factors: A dictionary of beta parameters for the risk factor equations:
            * ``fam_history``: A dictionary with the beta parameters for the family history
              odds ratio calculation. Must contain the keys ``β_fhx_0`` and ``β_fhx_age``.
            * ``abx``: A dictionary with the beta parameters for the antibiotic exposure
              odds ratio calculation. Must contain the keys ``β_abx_0``, ``β_abx_age``,
              and ``β_abx_dose``.
        min_year: The minimum year to consider for the calibration.

    Returns:
        A dictionary with the following keys:

            * ``prev_correction``: The correction for the asthma prevalence.
            * ``inc_correction``: The correction for the asthma incidence.
            * ``mean_diff_log_OR``: The mean difference in log odds ratio.
    """

    logger.info(f"Calibrating for year={year}, age={age}, sex={sex}")
    df = pd.Series(
        [year, sex, age, np.nan, 0, 0],
        index=["year", "sex", "age", "mean_diff_log_OR", "prev_correction", "inc_correction"]
    )

    risk_set = risk_factor_generator(year, age, sex, model_abx)

    if age > MAX_ABX_AGE:
        # group the all the antibiotic levels into one
        # only two risk levels: family history = {0, 1}
        risk_set = risk_set.groupby(["fam_history", "year", "sex", "age"]).agg(
            prob=("prob", "sum"),
            odds_ratio=("odds_ratio", "mean")
        ).reset_index()

    # target asthma prevalence from BC Ministry of Health
    asthma_prev_target = df_prevalence.loc[
        (df_prevalence["age"] == age) &
        (df_prevalence["year"] == year) &
        (df_prevalence["sex"] == sex)
    ]["prevalence"].iloc[0]

    # add target asthma prevalence to the risk set dataframe
    risk_set["prev"] = [asthma_prev_target] * risk_set.shape[0]

    # compute beta parameters for the prevalence correction
    asthma_prev_risk_factor_params = optimize_prevalence_β_parameters(
        asthma_prev_target=asthma_prev_target,
        odds_ratio_target=risk_set["odds_ratio"].to_list(),
        risk_factor_prev=risk_set["prob"].to_list()
    )

    # compute calibrated asthma prevalence using optimized beta parameters
    risk_set["calibrated_prev"] = compute_asthma_prevalence_λ(
        asthma_prev_risk_factor_params=asthma_prev_risk_factor_params,
        odds_ratio_target=risk_set["odds_ratio"].to_list(),
        risk_factor_prev=risk_set["prob"].to_list(),
        beta0=logit(asthma_prev_target)
    )

    df["prev_correction"] = -1 * get_asthma_prevalence_correction(
        asthma_prev_risk_factor_params, risk_set["prob"].to_list()
    )

    if year == 2000 or age == MIN_ASTHMA_AGE:
        if year > 2000 and age == MIN_ASTHMA_AGE:
            df["inc_correction"] = df["prev_correction"]
        return df

    else:
        # target asthma incidence for current year and age from BC Ministry of Health data
        risk_set["inc"] = [df_incidence.loc[
            (df_incidence["age"] == age) &
            (df_incidence["year"] == year) &
            (df_incidence["sex"] == sex)
        ]["incidence"].iloc[0]] * risk_set.shape[0]

        # target asthma prevalence for the previous year and age from BC Ministry of Health data
        past_asthma_prev_target = df_prevalence.loc[
            (df_prevalence["age"] == age - 1) &
            (df_prevalence["year"] == max(min_year, year - 1)) &
            (df_prevalence["sex"] == sex) 
        ]["prevalence"].iloc[0]

        past_risk_set = risk_factor_generator(
            year=max(min_year, year - 1),
            sex=sex,
            age=age - 1,
            model_abx=model_abx
        )

        ra_target = df_reassessment.loc[
            (df_reassessment["age"] == age) &
            (df_reassessment["year"] == year) &
            (df_reassessment["sex"] == sex)
        ]["ra"].iloc[0]

        if age > MAX_ABX_AGE + 1:
            # group the all the antibiotic levels into one
            # only two risk levels: family history = {0, 1}
            past_risk_set = past_risk_set.groupby(["fam_history"]).agg(
                prob=("prob", "sum"),
                odds_ratio=("odds_ratio", "mean")
            )
        elif age == MAX_ABX_AGE + 1:

            past_asthma_prev_risk_factor_params = optimize_prevalence_β_parameters(
                asthma_prev_target=past_asthma_prev_target,
                odds_ratio_target=past_risk_set["odds_ratio"].to_list(),
                risk_factor_prev=past_risk_set["prob"].to_list()
            )

            past_risk_set["calibrated_prev"] = compute_asthma_prevalence_λ(
                asthma_prev_risk_factor_params=past_asthma_prev_risk_factor_params,
                odds_ratio_target=past_risk_set["odds_ratio"].to_list(),
                risk_factor_prev=past_risk_set["prob"].to_list(),
                beta0=logit(past_asthma_prev_target)
            )
            past_risk_set["yes_asthma"] = past_risk_set.apply(
                lambda x: x["calibrated_prev"] * x["prob"], axis=1
            )
            past_risk_set["no_asthma"] = past_risk_set.apply(
                lambda x: (1 - x["calibrated_prev"]) * x["prob"], axis=1
            )
            odds_ratio_past = (
                past_risk_set.loc[past_risk_set["fam_history"] == 0, "no_asthma"].sum() *
                past_risk_set.loc[past_risk_set["fam_history"] == 1, "yes_asthma"].sum() /
                (past_risk_set.loc[past_risk_set["fam_history"] == 0, "yes_asthma"].sum() *
                past_risk_set.loc[past_risk_set["fam_history"] == 1, "no_asthma"].sum())
            )
            past_risk_set = past_risk_set.groupby(["fam_history"]).agg(
                prob=("prob", "sum")
            ).reset_index()
            past_risk_set["odds_ratio"] = [1, odds_ratio_past]


        inc_risk_set = risk_factor_generator(year, age, sex, model_abx)

        if age > MAX_ABX_AGE:
            # select only antibiotic dose = 0
            inc_risk_set = inc_risk_set.loc[inc_risk_set["n_abx"] == 0]


        inc_risk_set["odds_ratio"] = inc_risk_set.apply(
            lambda x: calculate_odds_ratio_risk_factors(
                fam_hist=x["fam_history"],
                age=x["age"],
                dose=x["n_abx"],
                β_risk_factors=β_risk_factors
            ),
            axis=1
        )
        
        if age <= MAX_ABX_AGE:
            inc_risk_set["prob"] = inc_risk_set.apply(
                lambda x: x["prob"] / inc_risk_set["prob"].sum(),
                axis=1
            )

    asthma_inc_correction, asthma_inc_calibrated, asthma_prev_calibrated_past = \
        inc_correction_calculator(
        asthma_inc_target=risk_set["inc"].iloc[0],
        asthma_prev_target_past=past_asthma_prev_target,
        odds_ratio_target_past=past_risk_set["odds_ratio"],
        risk_factor_prev_past=past_risk_set["prob"],
        risk_set=inc_risk_set
    )

    mean_diff_log_OR = compute_odds_ratio_difference(
        risk_factor_prev_past=past_risk_set["prob"],
        odds_ratio_target_past=past_risk_set["odds_ratio"],
        asthma_prev_calibrated_past=asthma_prev_calibrated_past,
        asthma_inc_calibrated=asthma_inc_calibrated,
        odds_ratio_target=risk_set["odds_ratio"].to_numpy(),
        ra_target=ra_target,
        misDx=0, # target misdiagnosis
        Dx=1, # target diagnosis
    )

    df["inc_correction"] = asthma_inc_correction
    df["mean_diff_log_OR"] = mean_diff_log_OR
    return df



def compute_mean_diff_log_OR(
    β_risk_factors_age: list[float],
    df: pd.DataFrame,
    model_abx: GLMResultsWrapper,
    df_incidence: pd.DataFrame,
    df_prevalence: pd.DataFrame,
    df_reassessment: pd.DataFrame,
    min_year: int = MIN_YEAR
) -> float:

    """Compute the mean difference in log odds ratio for the given model and data.

    Args:
        β_risk_factors_age: A list of two beta parameters, ``β_fhx_age`` and ``β_abx_age``.
        model_abx: The fitted ``Negative Binomial`` model for the number of courses of antibiotics.
        df_incidence: A dataframe with the incidence of asthma.
        df_prevalence: A dataframe with the prevalence of asthma.
        df_reassessment: A dataframe with the reassessment of asthma.
        baseline_year: The baseline year for the calibration.
        stabilization_year: The stabilization year for the calibration.
        max_age: The maximum age to consider for the calibration.

    Returns:
        The mean difference in log odds ratio for the given model and data.
    """
    β_risk_factors = {
        "fam_history": {
            "β_fhx_0": BETA_FHX_0,
            "β_fhx_age": β_risk_factors_age[0]
        },
        "abx": {
            "β_abx_0": BETA_ABX_0,
            "β_abx_age": β_risk_factors_age[1],
            "β_abx_dose": BETA_ABX_DOSE
        }
    }

    df["mean_log_diff_OR"] = df.apply(
        lambda x: calibrator(
            x["year"], x["sex"], x["age"], model_abx, df_incidence, df_prevalence,
            df_reassessment, β_risk_factors, min_year
        )["mean_diff_log_OR"],
        axis=1
    )
    return df["mean_log_diff_OR"].mean()


def beta_params_age_optimizer(
    model_abx: GLMResultsWrapper,
    df_incidence: pd.DataFrame,
    df_prevalence: pd.DataFrame,
    df_reassessment: pd.DataFrame,
    baseline_year: int = BASELINE_YEAR,
    stabilization_year: int = STABILIZATION_YEAR,
    max_age: int = MAX_AGE,
    min_year: int = MIN_YEAR,
    β_risk_factors_age: list[float] = [β_RISK_FACTORS["fam_history"][1], β_RISK_FACTORS["abx"][1]]
) -> None:
    """Optimize the risk factor beta parameters for the age terms.

    Args:
        model_abx: The fitted ``Negative Binomial`` model for the number of courses of antibiotics.
        df_incidence: A dataframe with the incidence of asthma.
        df_prevalence: A dataframe with the prevalence of asthma.
        df_reassessment: A dataframe with the reassessment of asthma.
        baseline_year: The baseline year for the calibration.
        stabilization_year: The stabilization year for the calibration.
        max_age: The maximum age to consider for the calibration.
        β_risk_factors_age: The initial beta parameters for the age terms in the
            risk factors equations.
    """

    df = pd.DataFrame(
        list(itertools.product(
            range(baseline_year, stabilization_year + 2),
            ["F", "M"],
            range(4, max_age + 1)
        )),
        columns=["year", "sex", "age"]
    )

    res = optimize.minimize(
        fun=compute_mean_diff_log_OR,
        x0=β_risk_factors_age,
        args=(
            df, model_abx, df_incidence, df_prevalence, df_reassessment, min_year
        ),
        method="BFGS",
        options={"maxiter": 1, "disp": True, "gtol": 1e-2}
    )

    with open(get_data_path("processed_data/occurrence_calibration_parameters.json"), "w") as f:
        json.dump(
            {
                "β_fhx_age": res.x[0],
                "β_abx_age": res.x[1],
            },
            f, indent=4
        )


def generate_occurrence_calibration_data(
    province: str = PROVINCE,
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR,
    baseline_year: int = BASELINE_YEAR,
    stabilization_year: int = STABILIZATION_YEAR,
    max_age: int = MAX_AGE,
    retrain_beta: bool = False
):
    """Generate the occurrence calibration data for the given province and year range.

    Args:
        province: The province to load data for.
        min_year: The minimum year to load data for.
        max_year: The maximum year to load data for.
        baseline_year: The baseline year for the calibration.
        stabilization_year: The stabilization year for the calibration.
        max_age: The maximum age to consider for the calibration.
        retrain_beta: If ``True``, re-run the fit for the ``β_risk_factors``. Otherwise, load
            the saved parameters from a ``json`` file.
    """

    df_incidence, df_prevalence = load_occurrence_data(
        province=province,
        min_year=min_year,
        max_year=max_year
    )

    df_reassessment = load_reassessment_data(province=province)

    model_abx: GLMResultsWrapper = generate_antibiotic_data(return_type="model") # type: ignore

    if retrain_beta:
        beta_params_age_optimizer(
            model_abx=model_abx,
            df_incidence=df_incidence,
            df_prevalence=df_prevalence,
            df_reassessment=df_reassessment,
            baseline_year=baseline_year,
            stabilization_year=stabilization_year,
            max_age=max_age
        )
    
    with open(get_data_path("processed_data/occurrence_calibration_parameters.json"), "r") as f:
        params = json.load(f)

    β_risk_factors = {
        "fam_history": {
            "β_fhx_0": BETA_FHX_0,
            "β_fhx_age": params["β_fhx_age"]
        },
        "abx": {
            "β_abx_0": BETA_ABX_0,
            "β_abx_age": params["β_abx_age"],
            "β_abx_dose": BETA_ABX_DOSE
        }
    }

    df_correction = pd.DataFrame(
        list(itertools.product(
            range(baseline_year - 1, stabilization_year + 1),
            ["F", "M"],
            range(3, max_age + 1)
        )),
        columns=["year", "sex", "age"]
    )
    
    df_correction = df_correction.apply(
        lambda x: calibrator(
            year=x["year"],
            sex=x["sex"],
            age=x["age"],
            model_abx=model_abx,
            df_incidence=df_incidence,
            df_prevalence=df_prevalence,
            df_reassessment=df_reassessment,
            β_risk_factors=β_risk_factors
        ), axis=1
    ).reset_index(drop=True)
    df_correction.sort_values(by=["sex", "age", "year"])

    df_correction_prevalence = df_correction[["year", "sex", "age", "prev_correction"]].copy()
    df_correction_prevalence.rename(columns={"prev_correction": "correction"}, inplace=True)
    df_correction_prevalence["type"] = ["prevalence"] * df_correction_prevalence.shape[0] 
    
    df_correction_incidence = df_correction[["year", "sex", "age", "inc_correction"]].copy()
    df_correction_incidence.rename(columns={"inc_correction": "correction"}, inplace=True)
    df_correction_incidence["type"] = ["incidence"] * df_correction_incidence.shape[0]

    df_correction = pd.concat(
        [df_correction_prevalence, df_correction_incidence],
        ignore_index=True
    )

    df_correction.to_csv(
        get_data_path("processed_data/asthma_occurrence_correction.csv"),
        index=False
    )


    
