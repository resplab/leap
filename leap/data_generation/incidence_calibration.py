import pandas as pd
import numpy as np
from leap.data_generation.utils import conv_2x2, ContingencyTable
from leap.data_generation.prevalence_calibration import optimize_prevalence_β_parameters, \
    compute_asthma_prevalence_λ, get_asthma_prevalence_correction
from leap.logger import get_logger
from typing import Tuple, Dict
from scipy.special import logit

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


def compute_contingency_tables(
    risk_factor_prob: list[float],
    odds_ratio_target: list[float],
    asthma_prev_calibrated: list[float],
    sample_size: float = 1e10
) -> Dict[str | int, ContingencyTable]:
    r"""Compute the contingency tables for the risk factors and asthma prevalence.

    This function computes the proportions of the population at different levels of family
    history and antibiotic exposure. The proportions are calculated based on the risk factors
    and the calibrated asthma prevalence. The function uses the ``metafor`` package to convert
    odds ratios into proportions.

    See: Bonett, D. G. (2007). Transforming odds ratios into correlations for
    meta-analytic research. American Psychologist, 62(3), 254-255.
    https://doi.org/10.1037/0003-066x.62.3.254

    Args:
        risk_factor_prob: A vector of the probabilities of the risk factor levels.
        odds_ratio_target: A vector of odds ratios for the risk factors.
        asthma_prev_calibrated: A vector of the calibrated asthma prevalence for each risk factor
            combination indexed by ``λ``.
        sample_size: The total population size to use for the calculations.

    Returns:
        A dictionary of dataframes representing the proportions of the population for different risk
        factor levels / combinations. For example, if we have the following risk factors:

        * family history: ``{0, 1}``
        * antibiotic exposure: ``{0, 1, 2, 3}``

        then we have ``2 * 4 = 8`` combinations. Each combination is called a ``risk factor level``,
        and is indexed by ``i`` (this corresponds to the index in the risk_set table). The first
        combination, ``i = 1`` is a special case; this is where there are no risk factors. We use
        this combination, referred to as the ``ref`` level, in the calculation of all the tables.
        
        Each dictionary entry contains a Pandas dataframe, with the following entries:

            * ``di``: proportion of population labelled as no asthma with no risk factors
            * ``ci``: proportion of population labelled as asthma with no risk factors
            * ``bi``: proportion of population labelled as no asthma with risk factors
            * ``ai``: proportion of population labelled as asthma with risk factors
    """

    contingency_tables = {}

    # calibrated asthma prevalence for population with no risk factors (λ = 0)
    asthma_prev_0 = asthma_prev_calibrated[0]

    # proportion of the population with no risk factors (λ = 0)
    risk_factor_prob_0 = risk_factor_prob[0]

    for λ in range(1, len(odds_ratio_target)):
        risk_factor_prob_λ = risk_factor_prob[λ]
        asthma_prev = asthma_prev_calibrated[λ]

        # probability of risk factor combination λ assuming only two possibilities: λ and 0
        risk_factor_prob_λ = risk_factor_prob_λ / (risk_factor_prob_λ + risk_factor_prob_0)

        # population with risk factor combination λ
        n1i = sample_size * risk_factor_prob_λ

        # expected prevalence of asthma
        n2i = sample_size * np.dot(
            [1 - risk_factor_prob_λ, risk_factor_prob_λ],
            [asthma_prev_0, asthma_prev]
        )
        table = conv_2x2(
            ori=odds_ratio_target[λ],
            ni=sample_size,
            n1i=n1i,
            n2i=n2i
        )

        # divide by sample size to get proportions
        table = table.apply(lambda x: x / sample_size)

        contingency_tables[λ] = table

    return contingency_tables



def compute_odds_ratio_difference(
    risk_factor_prob_past: list[float] | np.ndarray,
    odds_ratio_target_past: list[float] | np.ndarray,
    asthma_prev_calibrated_past: list[float] | np.ndarray,
    asthma_inc_calibrated: list[float] | np.ndarray,
    odds_ratio_target: list[float] | np.ndarray,
    ra_target: float = 1.0,
    misDx: float = 0,
    Dx: float = 1
) -> float:
    """Compute difference in odds ratios between the target and the calibrated asthma prevalence.

    Args:
        risk_factor_prob_past: A vector of the probabilities of the risk factor levels in the past.
        odds_ratio_target_past: A vector of odds ratios for the risk factors in the past.
        asthma_prev_calibrated_past: A vector of the calibrated asthma prevalence for each risk factor
            combination indexed by ``λ`` in the past.
        asthma_inc_calibrated: A vector of the calibrated asthma incidence for each risk factor
            combination indexed by ``λ`` in the current year.
        odds_ratio_target: A vector of odds ratios for the risk factors.
        ra_target: A value between 0 and 1 indicating the target reassessment.
        misDx: A numeric value representing the misdiagnosis rate.
        Dx: A numeric value representing the diagnosis rate.

    Returns:
        The sum of the difference in log odds ratios between the target and calibrated asthma
        prevalence for each risk factor level.
    """

    # for each odds ratio, we need to obtain the contingency table
    contingency_tables_past = compute_contingency_tables(
        risk_factor_prob=list(risk_factor_prob_past),
        odds_ratio_target=list(odds_ratio_target_past),
        asthma_prev_calibrated=list(asthma_prev_calibrated_past)
    )

    asthma_inc_calibrated_0 = asthma_inc_calibrated[0]
    total_diff_log_OR = 0

    for λ in range(1, len(odds_ratio_target)):
        asthma_inc = asthma_inc_calibrated[λ]
        
        # contingency table of the population with asthma from previous year t0
        a0 = contingency_tables_past[λ].loc["ai"].values[0] # proportion of population labelled as asthma with risk factors level λ
        b0 = contingency_tables_past[λ].loc["bi"].values[0] # proportion of population labelled as no asthma with risk factors level λ
        c0 = contingency_tables_past[λ].loc["ci"].values[0] # proportion of population labelled as asthma with no risk factors
        d0 = contingency_tables_past[λ].loc["di"].values[0] # proportion of population labelled as no asthma with no risk factors
        
        # contingency table current year t1, asthma reassessment
        # if ra=1, no reversibility
        a1_ra = a0 * ra_target # proportion of population who keep asthma diagnosis at risk factors level λ
        b1_ra = a0 * (1 - ra_target) # proportion of population who lose asthma diagnosis at risk factors level λ
        c1_ra = c0 * ra_target # proportion of population who keep asthma diagnosis with no risk factors
        d1_ra = c0 * (1 - ra_target) # proportion of population who lose asthma diagnosis with no risk factors
    
        # contingency table current year t1, asthma diagnosis
        # a1_dx: risk factors λ & asthma: got asthma and correctly Dx
        # t0 = no asthma diagnosis, t1 = no asthma diagnosis * misDx = has asthma +
        # t0 = no asthma diagnosis, t1 = asthma diagnosis * correct Dx = has asthma
        a1_dx = b0 * ((1 - asthma_inc) * misDx + asthma_inc * Dx)
        # b1_dx: risk factors λ & no asthma: did not get asthma and correctly Dx + got asthma but incorrectly Dx
        # t0 = no asthma diagnosis, t1 = no asthma diagnosis * correct Dx = no asthma +
        # t0 = no asthma diagnosis, t1 = asthma diagnosis * misDx = no asthma
        b1_dx = b0 * ((1 - asthma_inc) * (1 - misDx) + asthma_inc * (1 - Dx))
        # c1_dx: no risk factors & asthma: get asthma and correctly Dx + did not get asthma but misDx
        # t0 = no asthma diagnosis, t1 = no asthma diagnosis * misdiagnosis = has asthma + 
        # t0 = no asthma diagnosis, t1 = asthma diagnosis * correct Dx = has asthma
        c1_dx = d0 * ((1 - asthma_inc_calibrated_0) *  misDx + asthma_inc_calibrated_0 * Dx)
        # d1_dx: no risk factors & no asthma: 
        # t0 = no asthma diagnosis, t1 = no asthma diagnosis * correct Dx = no asthma + 
        # t0 = no asthma diagnosis, t1 = asthma diagnosis * misDx = no asthma
        d1_dx = d0 * ((1 - asthma_inc_calibrated_0) * (1 - misDx) + asthma_inc_calibrated_0 * (1 - Dx))

        # contingency table current year t1, asthma reassessment + diagnosis
        # objective: asthma prev OR
        # risk factors λ & asthma: proportion of population who either:
        # (a1_ra) keep asthma diagnosis from t0 - t1 or (a1_dx) get new asthma diagnosis at t1
        a1 = a1_ra + a1_dx
        # risk factors λ & no asthma: proportion of population who either:
        # (b1_ra) lose asthma diagnosis from t0 - t1 or (b1_dx) do not get new asthma diagnosis at t1
        b1 = b1_ra + b1_dx
        # no risk factors & asthma: proportion of population who either:
        # (c1_ra) keep asthma diagnosis from t0 - t1 or (c1_dx) get new asthma diagnosis at t1
        c1 = c1_ra + c1_dx
        # no risk factors & no asthma: proportion of population who either: 
        # (d1_ra) lose asthma diagnosis from t0 - t1 or (d1_dx) do not get new asthma diagnosis at t1
        d1 = d1_ra + d1_dx

        odds_ratio = (a1 * d1) / (b1 * c1)
        diff = np.abs(
            np.log(odds_ratio_target[λ]) - (np.log(odds_ratio))
        )
        total_diff_log_OR += diff

    return total_diff_log_OR



def inc_correction_calculator(
    asthma_inc_target: float,
    asthma_prev_target_past: float,
    odds_ratio_target_past: np.ndarray,
    risk_factor_prev_past: np.ndarray,
    risk_set: pd.DataFrame
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate the correction for asthma incidence based on risk factors.

    Args:
        asthma_inc_target: The target incidence of asthma.
        asthma_prev_target_past: The target prevalence of asthma in the previous year.
        odds_ratio_target_past: A vector of odds ratios for the risk factors in the previous year.
        risk_factor_prev_past: A vector of the prevalence of the risk factors in the previous year.
        risk_set: A data frame containing the risk factors and their corresponding odds ratios.

    Returns:
        A tuple containing three entries:
        * ``asthma_inc_correction``: the calibrated asthma incidence correction.
        * ``asthma_inc_calibrated``: the calibrated asthma incidence.
        * ``asthma_prev_calibrated_past``: the calibrated asthma prevalence for the previous year.
    """

    # target asthma incidence from the BC Ministry of Health data model
    β_0 = logit(asthma_inc_target)
    
    # asthma prevalence ~ risk factor parameters for the previous year
    asthma_prev_risk_factor_params_past = optimize_prevalence_β_parameters(
        asthma_prev_target=asthma_prev_target_past,
        odds_ratio_target=odds_ratio_target_past,
        risk_factor_prev=risk_factor_prev_past
    )

    # calibrated asthma prevalence for the previous year
    asthma_prev_calibrated_past = compute_asthma_prevalence_λ(
        asthma_prev_risk_factor_params=asthma_prev_risk_factor_params_past,
        odds_ratio_target=list(odds_ratio_target_past),
        risk_factor_prev=list(risk_factor_prev_past),
        beta0=logit(asthma_prev_target_past)
    )

    # joint probability of risk factors and no asthma in the previous year
    # P(λ, A = 0) = P(A = 0 | λ) * P(λ) = (1 - P(A = 1 | λ)) * P(λ)
    risk_factor_prev_past_no_asthma = (1 - asthma_prev_calibrated_past) * risk_factor_prev_past
    # normalize
    risk_factor_prev_past_no_asthma = risk_factor_prev_past_no_asthma / np.sum(risk_factor_prev_past_no_asthma)
    
    # asthma prevalence ~ risk factor parameters for incidence
    asthma_prev_risk_factor_params = optimize_prevalence_β_parameters(
        asthma_prev_target=asthma_inc_target,
        odds_ratio_target=risk_set["odds_ratio"].to_list(),
        risk_factor_prev=list(risk_factor_prev_past_no_asthma)
    )

    # asthma incidence correction term for the current year
    asthma_inc_correction = get_asthma_prevalence_correction(
        asthma_prev_risk_factor_params, risk_factor_prev_past_no_asthma
    )

    # calibrated asthma incidence for the current year
    asthma_inc_calibrated = compute_asthma_prevalence_λ(
        asthma_prev_risk_factor_params=asthma_prev_risk_factor_params,
        odds_ratio_target=risk_set["odds_ratio"].to_list(),
        risk_factor_prev=list(risk_factor_prev_past_no_asthma),
        beta0=β_0
    )

    return asthma_inc_correction, asthma_inc_calibrated, asthma_prev_calibrated_past




