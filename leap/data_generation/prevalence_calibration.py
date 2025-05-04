import pandas as pd
import numpy as np
from scipy import optimize
from leap.logger import get_logger
from scipy.stats import logistic
from scipy.special import logit

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


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






