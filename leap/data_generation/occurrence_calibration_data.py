import pathlib
import pandas as pd
import numpy as np
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from scipy import optimize
import itertools
from leap.utils import get_data_path
from leap.data_generation.utils import heaviside, conv_2x2
from leap.data_generation.antibiotic_data import get_predicted_abx_data
from leap.logger import get_logger
from typing import Tuple
from scipy.stats import logistic, nbinom
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

    df_abx_or = df_abx_or.melt(id_vars=["age"], var_name="n_abx", value_name="odds_ratio")
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
    df_abx_exposure.loc[df_abx_exposure["n_abx"] == 3, "prob"] = df_abx_exposure.loc[
        df_abx_exposure["n_abx"] >= 3, "prob"
    ].sum()
    df_abx_exposure = df_abx_exposure[df_abx_exposure["n_abx"] <= 3]
    df_abx_exposure.rename(columns={"prob": "prob_abx"}, inplace=True)


    # select the given age if <= 5, otherwise select age == 5
    df_fam_history_or_age = df_fam_history_or.loc[df_fam_history_or["age"] == min(age, 5)]
    df_fam_history_or_age = df_fam_history_or_age[["fam_history", "odds_ratio"]]
    df_fam_history_or_age.rename(columns={"odds_ratio": "odds_ratio_fam_hist"}, inplace=True)
  
    # select the given age if <= 8, otherwise select age == 8
    # filter out abx_exposure > 3
    df_abx_or_age = df_abx_or.loc[df_abx_or["age"] == min(age, 8)]
    df_abx_or_age = df_abx_or_age[["n_abx", "odds_ratio"]]
    df_abx_or_age.rename(columns={"odds_ratio": "odds_ratio_abx"}, inplace=True)
    df_abx_or_age = df_abx_or_age.loc[df_abx_or_age["n_abx"] <= 3]
       
    df_risk_factors = pd.DataFrame(
        list(itertools.product([0, 1], range(0, 4))),
        columns=["fam_history", "n_abx"]
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
    p_fam_distribution: pd.DataFrame,
    df_fam_history_or: pd.DataFrame,
    df_abx_or: pd.DataFrame,
    df_incidence: pd.DataFrame,
    df_prevalence: pd.DataFrame,
    df_reassessment: pd.DataFrame,
    inc_beta_params: list[float] | dict = [0.3766256, BETA_ABX_AGE],
    min_year: int = MIN_YEAR
) -> dict:
    """Compute the loss function given the effects of risk factors in the incidence equation, for
    each year, age, and sex.
)

    Args:
        year: The integer year.
        age: The age in years.
        sex: One of ``M`` = male, ``F`` = female.
        model_abx: The fitted ``Negative Binomial`` model for the number of courses of antibiotics.
        p_fam_distribution: A dataframe with the probability of family history of asthma, given
            that the person has asthma. Contains two columns:
            * ``fam_history (int)``: (0 or 1) Whether or not there is a family history of asthma.
              ``1`` = one or more parents has asthma, ``0`` = no parents have asthma.
            * ``prob_fam (float)``: The probability that one or more parents has asthma, given
              that the person has asthma.
        df_fam_history_or: A dataframe with the odds ratio of family history of asthma, given
            the age of the person. Contains three columns:
            * ``age (int)``: age in years, one of ``{3, 4, 5}``.
            * ``fam_history (int)``: (0 or 1) Whether or not there is a family history of asthma.
              ``1`` = one or more parents has asthma, ``0`` = no parents have asthma.
            * ``odds_ratio (float)``: The odds ratio for having asthma given family history status.
        df_abx_or: A dataframe with the odds ratio of antibiotic exposure, given
            the age of the person. Contains three columns:
            * ``age (int)``: age in years, one of ``{3, 4, 5}``.
            * ``n_abx (int)``: The number of courses of antibiotics taken during the first year of
              life, an integer in ``[0, 5]``, where ``5`` indicates 5 or more courses.
            * ``odds_ratio (float)``: The odds ratio for having asthma given ``n_abx`` courses
              of antibiotics.
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
        inc_beta_params: A list or dictionary of parameters for the incidence equation.
        min_year: The minimum year to consider for the calibration.

    Returns:
        A dictionary with the following keys:

            * ``prev_correction``: The correction for the asthma prevalence.
            * ``inc_correction``: The correction for the asthma incidence.
            * ``mean_diff_log_OR``: The mean difference in log odds ratio.
    """

    logger.info(f"Calibrating for year={year}, age={age}, sex={sex}")

    if type(inc_beta_params) != dict:
        inc_beta_params = {
            "fam_history":  [np.log(OR_ASTHMA_AGE_3), inc_beta_params[0]],
            "abx": [BETA_ABX_0, inc_beta_params[1], BETA_ABX_DOSE]
        }


    risk_set = risk_factor_generator(
        year, age, sex, model_abx, p_fam_distribution, df_fam_history_or, df_abx_or
    )

    if age > 7:
        # group the all the antibiotic levels into one
        # only two risk levels: family history = {0, 1}
        risk_set = risk_set.groupby(["fam_history", "year", "sex", "age"]).agg(
            prob=("prob", "sum"),
            odds_ratio=("odds_ratio", "mean")
        ).reset_index()

    # target marginal asthma prevalence
    asthma_prev_target = df_prevalence.loc[
        (df_prevalence["age"] == age) &
        (df_prevalence["year"] == year) &
        (df_prevalence["sex"] == sex)
    ]["prevalence"].iloc[0]

    asthma_prev_risk_factor_params = prev_calibrator(
        asthma_prev_target=asthma_prev_target,
        odds_ratio_target=risk_set["odds_ratio"],
        risk_factor_prev=risk_set["prob"]
    )

    risk_set["prev"] = [asthma_prev_target] * risk_set.shape[0]
    risk_set["calibrated_prev"] = logistic.cdf(
        logit([asthma_prev_target] * risk_set.shape[0]) +
        np.log(risk_set["odds_ratio"].to_numpy()) -
        [np.dot(risk_set["prob"].iloc[1:].to_numpy(), asthma_prev_risk_factor_params)] * risk_set.shape[0]
    )

    if year == 2000 or age == 3:
        return {
            "prev_correction": -np.dot(risk_set["prob"].iloc[1:], asthma_prev_risk_factor_params),
            "inc_correction": None,
            "mean_diff_log_OR": None
        }

    else:
        risk_set["inc"] = [df_incidence.loc[
            (df_incidence["age"] == age) &
            (df_incidence["year"] == year) &
            (df_incidence["sex"] == sex)
        ]["incidence"].iloc[0]] * risk_set.shape[0]

        # target marginal asthma prevalence for the previous year and age
        past_asthma_prev_target = df_prevalence.loc[
            (df_prevalence["age"] == age - 1) &
            (df_prevalence["year"] == max(min_year, year - 1)) &
            (df_prevalence["sex"] == sex) 
        ]["prevalence"].iloc[0]

        past_risk_set = risk_factor_generator(
            year=max(min_year, year - 1),
            sex=sex,
            age=age - 1,
            model_abx=model_abx,
            p_fam_distribution=p_fam_distribution,
            df_fam_history_or=df_fam_history_or,
            df_abx_or=df_abx_or
        )

        ra_target = df_reassessment.loc[
            (df_reassessment["age"] == age) &
            (df_reassessment["year"] == year) &
            (df_reassessment["sex"] == sex)
        ]["ra"].iloc[0]

        if age > 8:
            # group the all the antibiotic levels into one
            # only two risk levels: family history = {0, 1}
            past_risk_set = past_risk_set.groupby(["fam_history"]).agg(
                prob=("prob", "sum"),
                odds_ratio=("odds_ratio", "mean")
            )
        elif age == 8:

            ttt_asthma_prev_risk_factor_params = prev_calibrator(
                asthma_prev_target=past_asthma_prev_target,
                odds_ratio_target=past_risk_set["odds_ratio"],
                risk_factor_prev=past_risk_set["prob"]
            )

            past_risk_set["calibrated_prev"] = logistic.cdf(
                [logit(past_asthma_prev_target)] * past_risk_set.shape[0] + 
                np.log(past_risk_set["odds_ratio"]) -
                [np.dot(past_risk_set["prob"].iloc[1:], ttt_asthma_prev_risk_factor_params)] * past_risk_set.shape[0]
            )
            tmp_look = past_risk_set.copy()
            tmp_look["yes_asthma"] = tmp_look.apply(
                lambda x: x["calibrated_prev"] * x["prob"], axis=1
            )
            tmp_look["no_asthma"] = tmp_look.apply(
                lambda x: (1 - x["calibrated_prev"]) * x["prob"], axis=1
            )
            past_tmp_OR = (
                tmp_look.loc[tmp_look["fam_history"] == 0, "no_asthma"].sum() *
                tmp_look.loc[tmp_look["fam_history"] == 1, "yes_asthma"].sum() /
                (tmp_look.loc[tmp_look["fam_history"] == 0, "yes_asthma"].sum() *
                tmp_look.loc[tmp_look["fam_history"] == 1, "no_asthma"].sum())
            )
            past_risk_set = past_risk_set.groupby(["fam_history"]).agg(
                prob=("prob", "sum")
            ).reset_index()
            past_risk_set["odds_ratio"] = [1, past_tmp_OR]


        inc_risk_set = risk_factor_generator(
            year, age, sex, model_abx, p_fam_distribution, df_fam_history_or, df_abx_or
        )
        # drop the odds_ratio column
        inc_risk_set = inc_risk_set[["fam_history", "n_abx", "year", "sex", "age", "prob"]]

        if age > 7:
            # select only antibiotic level 0
            inc_risk_set = inc_risk_set.loc[inc_risk_set["n_abx"] == 0]


        inc_risk_set["odds_ratio"] = inc_risk_set.apply(
            lambda x: OR_risk_factor_calculator(
                fam_hist=x["fam_history"],
                age=x["age"],
                dose=x["n_abx"],
                family_history_params=inc_beta_params["fam_history"],
                abx_params=inc_beta_params["abx"]
            ),
            axis=1
        )
        
        if age <= 7:
            inc_risk_set["prob"] = inc_risk_set.apply(
                lambda x: x["prob"] / inc_risk_set["prob"].sum(),
                axis=1
            )

    asthma_inc_correction, mean_diff_log_OR = inc_correction_calculator(
        asthma_inc_target=risk_set["inc"].iloc[0],
        asthma_prev_target_past=past_asthma_prev_target,
        past_target_OR=past_risk_set["odds_ratio"],
        target_OR=risk_set["odds_ratio"],
        risk_factor_prev_past=past_risk_set["prob"],
        risk_set=inc_risk_set,
        ra_target=ra_target,
        misDx=0, # target misdiagnosis
        Dx=1, # target diagnosis
    )

    return {
        "prev_correction": -np.dot(risk_set["prob"].iloc[1:], asthma_prev_risk_factor_params),
        "inc_correction": asthma_inc_correction,
        "mean_diff_log_OR": mean_diff_log_OR
    }

def generate_occurrence_calibration_data(
    province: str = PROVINCE,
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR,
    baseline_year: int = BASELINE_YEAR,
    stabilization_year: int = STABILIZATION_YEAR,
    max_age: int = MAX_AGE
):
    """Generate the occurrence calibration data for the given province and year range.

    Args:
        province: The province to load data for.
        min_year: The minimum year to load data for.
        max_year: The maximum year to load data for.
        baseline_year: The baseline year for the calibration.
        stabilization_year: The stabilization year for the calibration.
        max_age: The maximum age to consider for the calibration.
    """

    df_incidence, df_prevalence = load_occurrence_data(
        province=province,
        min_year=min_year,
        max_year=max_year
    )

    df_reassessment = load_reassessment_data(province=province)

    p_fam_distribution = pd.DataFrame({
        "fam_history": [0, 1],
        "prob_fam": [1 - PROB_FAM_HIST, PROB_FAM_HIST]
    })

    df_fam_history_or = load_family_history_data()
    df_abx_or = load_abx_exposure_data()

    model_abx = generate_antibiotic_data(return_type="model")

    optimized_inc_beta = load_optimized_beta_params(
        stabilization_year=stabilization_year, baseline_year=baseline_year, max_age=max_age
    )

    df_correction = pd.DataFrame(
        list(itertools.product(
            range(baseline_year - 1, stabilization_year + 1),
            ["F", "M"],
            range(3, max_age + 1)
        )),
        columns=["year", "sex", "age"]
    )
    
    df_correction = df_correction.apply(
        lambda x: calculate_correction(
            year=x["year"],
            sex=x["sex"],
            age=x["age"],
            model_abx=model_abx,
            p_fam_distribution=p_fam_distribution,
            df_fam_history_or=df_fam_history_or,
            df_abx_or=df_abx_or,
            df_incidence=df_incidence,
            df_prevalence=df_prevalence,
            df_reassessment=df_reassessment,
            inc_beta_params=optimized_inc_beta
        ), axis=1
    ).reset_index(drop=True)

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



def compute_asthma_prev_risk_factors(
    asthma_prev_risk_factor_params: list[float],
    odds_ratio_target: list[float],
    risk_factor_prev: list[float],
    beta0: float
) -> float:
    """Compute the asthma prevalence based on the risk factors and the parameters provided.

    Args:
        asthma_prev_risk_factor_params: A vector of parameters for the risk factors, with
            shape ``(n - 1, 1)``.
        odds_ratio_target: A vector of odds ratios between the risk factors and asthma, with
            shape ``(n, 1)``.
        risk_factor_prev: A vector of the prevalence of the risk factor levels, with shape
            ``(n, 1)``.
        beta0: The intercept of the logistic regression model.
    
    Returns:
        The calibrated asthma prevalence.
    """

    n = len(odds_ratio_target)
    # asthma_prev_x: asthma prevalence at risk factor level x
    asthma_prev_x = logistic.cdf(
        beta0 * np.ones(n) + 
        np.log(odds_ratio_target) - 
        np.dot(risk_factor_prev[1:], asthma_prev_risk_factor_params) * np.ones(n)
    )
    return(np.dot(asthma_prev_x, risk_factor_prev))


def compute_asthma_prevalence_difference(
    asthma_prev_risk_factor_params: list[float],
    odds_ratio_target: list[float],
    risk_factor_prev: list[float],
    beta0: float,
    asthma_prev_target: float
) -> float:
    """Compute the absolute difference between the calibrated and target asthma prevalence.

    Args:
        asthma_prev_risk_factor_params: A vector of parameters for the risk factors, with
            shape ``(n - 1, 1)``.
        odds_ratio_target: A vector of odds ratios between the risk factors and asthma, with
            shape ``(n, 1)``.
        risk_factor_prev: A vector of the prevalence of the risk factor levels, with shape
            ``(n, 1)``.
        beta0: The intercept of the logistic regression model.
        asthma_prev_target: The target prevalence of asthma.

    Returns:
        The absolute difference between the calibrated and target asthma prevalence.
    """

    asthma_prev_calibrated = compute_asthma_prev_risk_factors(
        asthma_prev_risk_factor_params, odds_ratio_target, risk_factor_prev, beta0
    )
    return np.abs(asthma_prev_calibrated - asthma_prev_target)


def prev_calibrator(
    asthma_prev_target: float,
    odds_ratio_target: list[float],
    risk_factor_prev: list[float],
    beta0: float | None = None,
    verbose: bool = False
) -> list[float]:
    """Calibrate asthma prevalence based on the target prevalence and odds ratios of risk factors.

    Args:
        asthma_prev_target: The target prevalence of asthma.
        odds_ratio_target: A vector of odds ratios for the risk factors, with shape ``(n, 1)``.
        risk_factor_prev: A vector of the prevalence of the risk factors, with shape ``(n, 1)``.
        beta0: The intercept of the logistic regression model. If ``None``, it is set to
            the ``logit`` of the target prevalence.
        verbose: A boolean indicating if the trace should be printed.

    Returns:
        A vector of the calibrated asthma prevalence for each risk factor level.
    """

    if beta0 is None:
        beta0 = logit(asthma_prev_target)

    n_params = len(odds_ratio_target) - 1

    res = optimize.minimize(
        fun=compute_asthma_prevalence_difference,
        x0=np.zeros(n_params),
        args=(odds_ratio_target, risk_factor_prev, beta0, asthma_prev_target),
        method="BFGS",
        tol=1e-15,
        hess=True,
        options={"maxiter": 10000, "disp": verbose}
    )

    return res.x



def compute_contingency_table(
    risk_factor_prev: list[float],
    odds_ratio_target: list[float],
    asthma_prev_calibrated: list[float]
) -> list[pd.DataFrame]:
    r"""Compute the contingency table for the risk factors and asthma prevalence.

    This function computes the proportions of the population at different levels of family
    history and antibiotic exposure. The proportions are calculated based on the risk factors
    and the calibrated asthma prevalence. The function uses the ``metafor`` package to convert
    odds ratios into proportions.

    See: Bonett, D. G. (2007). Transforming odds ratios into correlations for
    meta-analytic research. American Psychologist, 62(3), 254-255.
    https://doi.org/10.1037/0003-066x.62.3.254

    Args:
        risk_factor_prev: A vector of the prevalence of the risk factor levels.
        odds_ratio_target: A vector of odds ratios for the risk factors.
        asthma_prev_calibrated: A vector of the calibrated asthma prevalence.

    Returns:
        A list of vectors representing the proportions of the population for different risk
        factor levels / combinations. For example, if we have the following risk factors:

        * family history: ``{0, 1}``
        * antibiotic exposure: ``{0, 1, 2, 3}``

        then we have ``2 * 4 = 8`` combinations. Each combination is called a ``risk factor level``,
        and is indexed by ``i`` (this corresponds to the index in the risk_set table). The first
        combination, ``i = 1`` is a special case; this is where there are no risk factors. We use
        this combination, referred to as the ``ref`` level, in the calculation of all the tables.
        
        Each list entry contains a Pandas dataframe, with the following entries:

            * ``di``: proportion of population labelled as no asthma with no risk factors
            * ``ci``: proportion of population labelled as asthma with no risk factors
            * ``bi``: proportion of population labelled as no asthma with risk factors
            * ``ai``: proportion of population labelled as asthma with risk factors
    """

    contingency_tables = []

    asthma_prev_ref = asthma_prev_calibrated[0]
    risk_factor_prev_ref = risk_factor_prev[0]

    for i in range(len(odds_ratio_target)):
        risk_factor_prev_i = risk_factor_prev[i]
        asthma_prev = asthma_prev_calibrated[i]

        # prevalence of risk factor combination i
        risk_factor_prev_i = risk_factor_prev_i / (risk_factor_prev_i + risk_factor_prev_ref)

        sample_size = 1e10
        # prevalence of risk factor combination i
        n1i = sample_size * risk_factor_prev_i
        # prevalence of asthma
        n2i = sample_size * np.dot(
            [1 - risk_factor_prev_i, risk_factor_prev_i],
            [asthma_prev_ref, asthma_prev]
        )
        table = conv_2x2(
            ori=odds_ratio_target[i],
            ni=sample_size,
            n1i=n1i,
            n2i=n2i
        )

        # divide by sample size to get proportions
        table["values"] = table["values"].apply(lambda x: x / sample_size)

        contingency_tables.append(table)

    return contingency_tables
