import pandas as pd
import numpy as np
from leap.data_generation.utils import conv_2x2
from leap.logger import get_logger
from typing import Tuple, Dict
from scipy.stats import logistic
from scipy.special import logit

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


def compute_contingency_table(
    risk_factor_prev: list[float],
    odds_ratio_target: list[float],
    asthma_prev_calibrated: list[float]
) -> Dict[str | int, pd.DataFrame]:
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
        A dictionary of dataframes representing the proportions of the population for different risk
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

    contingency_tables = {}

    asthma_prev_ref = asthma_prev_calibrated[0]
    risk_factor_prev_ref = risk_factor_prev[0]

    for i in range(1, len(odds_ratio_target)):
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

        contingency_tables[i] = table

    return contingency_tables



