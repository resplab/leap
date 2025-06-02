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
    """Compute the contingency tables for the risk factors and asthma prevalence.

    This function computes the proportions of the population at different levels of family
    history and antibiotic exposure. The proportions are calculated based on the risk factors
    and the calibrated asthma prevalence. The function uses the ``metafor`` package to convert
    odds ratios into proportions.

    See :cite:`bonett2007` for more details.

    Args:
        risk_factor_prob: A vector of the probabilities of the risk factor levels.
        odds_ratio_target: A vector of odds ratios for the risk factors.
        asthma_prev_calibrated: A vector of the calibrated asthma prevalence for each risk factor
            combination indexed by ``λ``.
        sample_size: The total population size to use for the calculations.

    Returns:
        A dictionary of ``ContingencyTables`` representing the proportions of the population for
        different risk factor levels / combinations. For example, if we have the following
        risk factors:

        * family history: ``{0, 1}``
        * antibiotic exposure: ``{0, 1, 2, 3}``

        then we have ``2 * 4 = 8`` combinations. Each combination is called a ``risk factor level``,
        and is indexed by ``λ`` (this corresponds to the index in the risk_set table). The first
        combination, ``λ = 0`` is a special case; this is where there are no risk factors.
        
        Each dictionary entry contains a ``ContingencyTable``, with the following entries:

        * ``a``: proportion of population labelled as asthma with risk factors ``λ``
        * ``b``: proportion of population labelled as no asthma with risk factors ``λ``
        * ``c``: proportion of population labelled as asthma with no risk factors 
        * ``d``: proportion of population labelled as no asthma with no risk factors

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


def compute_odds_ratio(
    contingency_table_past: ContingencyTable,
    asthma_incidence_0: float,
    asthma_incidence_λ: float,
    ra_target: float = 1.0,
    mis_dx: float = 0,
    dx: float = 1
) -> float:
    """Compute the odds ratio for risk factor combination ``λ``.

    Args:
        contingency_table_past: A ``ContingencyTable`` object representing the proportions of the
            population for different risk factor levels in the past.
        asthma_incidence_0: The calibrated asthma incidence for the risk factor combination with no
            risk factors (``λ = 0``) in the current year.
        asthma_incidence_λ: The calibrated asthma incidence for the risk factor combination ``λ`` in
            the current year.
        ra_target: The probability that an individual previously diagnosed with asthma will
            maintain their asthma diagnosis after reassessment in the current year.
            Range: ``[0, 1]``.
        mis_dx: A numeric value representing the misdiagnosis rate.
        dx: A numeric value representing the diagnosis rate.

    Returns:
        The odds ratio for risk factor combination ``λ`` in the current year.
    """
    
    # contingency table of the population with asthma from previous year t0
    a0 = contingency_table_past.a # proportion of population labelled as asthma with risk factors level λ
    b0 = contingency_table_past.b # proportion of population labelled as no asthma with risk factors level λ
    c0 = contingency_table_past.c # proportion of population labelled as asthma with no risk factors
    d0 = contingency_table_past.d # proportion of population labelled as no asthma with no risk factors
    
    # contingency table current year t1, asthma reassessment
    # if ra=1, no reversibility
    a1_ra = a0 * ra_target # proportion of population who keep asthma diagnosis at risk factors level λ
    b1_ra = a0 * (1 - ra_target) # proportion of population who lose asthma diagnosis at risk factors level λ
    c1_ra = c0 * ra_target # proportion of population who keep asthma diagnosis with no risk factors
    d1_ra = c0 * (1 - ra_target) # proportion of population who lose asthma diagnosis with no risk factors

    # contingency table current year t1, asthma diagnosis
    # a1_dx: risk factors λ & asthma: got asthma and correctly dx
    # t0 = no asthma diagnosis, t1 = no asthma diagnosis * mis_dx = has asthma +
    # t0 = no asthma diagnosis, t1 = asthma diagnosis * correct dx = has asthma
    a1_dx = b0 * ((1 - asthma_incidence_λ) * mis_dx + asthma_incidence_λ * dx)
    # b1_dx: risk factors λ & no asthma: did not get asthma and correctly dx + got asthma but incorrectly dx
    # t0 = no asthma diagnosis, t1 = no asthma diagnosis * correct dx = no asthma +
    # t0 = no asthma diagnosis, t1 = asthma diagnosis * mis_dx = no asthma
    b1_dx = b0 * ((1 - asthma_incidence_λ) * (1 - mis_dx) + asthma_incidence_λ * (1 - dx))
    # c1_dx: no risk factors & asthma: get asthma and correctly dx + did not get asthma but mis_dx
    # t0 = no asthma diagnosis, t1 = no asthma diagnosis * misdiagnosis = has asthma + 
    # t0 = no asthma diagnosis, t1 = asthma diagnosis * correct dx = has asthma
    c1_dx = d0 * ((1 - asthma_incidence_0) *  mis_dx + asthma_incidence_0 * dx)
    # d1_dx: no risk factors & no asthma: 
    # t0 = no asthma diagnosis, t1 = no asthma diagnosis * correct dx = no asthma + 
    # t0 = no asthma diagnosis, t1 = asthma diagnosis * mis_dx = no asthma
    d1_dx = d0 * ((1 - asthma_incidence_0) * (1 - mis_dx) + asthma_incidence_0 * (1 - dx))

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

    return odds_ratio


def compute_odds_ratio_difference(
    risk_factor_prob_past: list[float] | np.ndarray,
    odds_ratio_target_past: list[float] | np.ndarray,
    asthma_prev_calibrated_past: list[float] | np.ndarray,
    asthma_inc_calibrated: list[float] | np.ndarray,
    odds_ratio_target: list[float] | np.ndarray,
    ra_target: float = 1.0,
    mis_dx: float = 0,
    dx: float = 1
) -> float:
    """Compute difference in odds ratios between the target and the calibrated asthma incidence.

    Args:
        risk_factor_prob_past: A vector of the probabilities of the risk factor levels in the past.
        odds_ratio_target_past: A vector of odds ratios for the risk factors in the past.
        asthma_prev_calibrated_past: A vector of the calibrated asthma prevalence for each risk factor
            combination indexed by ``λ`` in the past.
        asthma_inc_calibrated: A vector of the calibrated asthma incidence for each risk factor
            combination indexed by ``λ`` in the current year.
        odds_ratio_target: A vector of odds ratios for the risk factors.
        ra_target: The probability that an individual previously diagnosed with asthma will
            maintain their asthma diagnosis after reassessment in the current year.
            Range: ``[0, 1]``.
        mis_dx: A numeric value representing the misdiagnosis rate.
        dx: A numeric value representing the diagnosis rate.

    Returns:
        The sum of the difference in log odds ratios between the target and calibrated asthma
        incidence for each risk factor level.
    """

    # for each odds ratio, we need to obtain the contingency table
    contingency_tables_past = compute_contingency_tables(
        risk_factor_prob=list(risk_factor_prob_past),
        odds_ratio_target=list(odds_ratio_target_past),
        asthma_prev_calibrated=list(asthma_prev_calibrated_past)
    )

    odds_ratios = [
        compute_odds_ratio(
            contingency_table_past=contingency_tables_past[λ],
            asthma_incidence_0=asthma_inc_calibrated[0],
            asthma_incidence_λ=asthma_inc_calibrated[λ],
            ra_target=ra_target,
            mis_dx=mis_dx,
            dx=dx
        )
        for λ in range(1, len(odds_ratio_target))
    ]
    # compute the difference in log odds ratios
    diff = np.abs(
        np.log(odds_ratio_target[1:]) - np.log(odds_ratios)
    )

    return np.sum(diff)



def inc_correction_calculator(
    asthma_inc_target: float,
    asthma_prev_target_past: float,
    odds_ratio_target_past: np.ndarray,
    risk_factor_prob_past: np.ndarray,
    risk_set: pd.DataFrame
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate the correction for asthma incidence based on risk factors.

    Args:
        asthma_inc_target: The target incidence of asthma.
        asthma_prev_target_past: The target prevalence of asthma in the previous year.
        odds_ratio_target_past: A vector of odds ratios for the risk factors in the previous year.
        risk_factor_prob_past: A vector of the prevalence of the risk factors in the previous year.
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
        risk_factor_prob=risk_factor_prob_past
    )

    # calibrated asthma prevalence for the previous year
    asthma_prev_calibrated_past = compute_asthma_prevalence_λ(
        asthma_prev_risk_factor_params=asthma_prev_risk_factor_params_past,
        odds_ratio_target=list(odds_ratio_target_past),
        risk_factor_prob=list(risk_factor_prob_past),
        β_0=logit(asthma_prev_target_past)
    )

    # joint probability of risk factors and no asthma in the previous year
    # P(λ, A = 0) = P(A = 0 | λ) * P(λ) = (1 - P(A = 1 | λ)) * P(λ)
    risk_factor_prob_past_no_asthma = (1 - asthma_prev_calibrated_past) * risk_factor_prob_past
    # normalize
    risk_factor_prob_past_no_asthma = risk_factor_prob_past_no_asthma / np.sum(risk_factor_prob_past_no_asthma)
    
    # asthma prevalence ~ risk factor parameters for incidence
    asthma_prev_risk_factor_params = optimize_prevalence_β_parameters(
        asthma_prev_target=asthma_inc_target,
        odds_ratio_target=risk_set["odds_ratio"].to_list(),
        risk_factor_prob=list(risk_factor_prob_past_no_asthma)
    )

    # asthma incidence correction term for the current year
    asthma_inc_correction = get_asthma_prevalence_correction(
        asthma_prev_risk_factor_params, risk_factor_prob_past_no_asthma
    )

    # calibrated asthma incidence for the current year
    asthma_inc_calibrated = compute_asthma_prevalence_λ(
        asthma_prev_risk_factor_params=asthma_prev_risk_factor_params,
        odds_ratio_target=risk_set["odds_ratio"].to_list(),
        risk_factor_prob=list(risk_factor_prob_past_no_asthma),
        β_0=β_0
    )

    return asthma_inc_correction, asthma_inc_calibrated, asthma_prev_calibrated_past




