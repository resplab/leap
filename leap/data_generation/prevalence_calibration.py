import pandas as pd
import numpy as np
from scipy import optimize
from leap.logger import get_logger
from scipy.stats import logistic
from scipy.special import logit

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


def get_asthma_prevalence_correction(
    asthma_prev_risk_factor_params: list[float],
    risk_factor_prob: list[float]
) -> float:
    r"""Compute the correction term for asthma prevalence.

    .. math::

        \alpha = \sum_{\lambda=1}^{n} p(\lambda) \cdot \beta_{\lambda}

    where:

    * :math:`\alpha` is the correction term for the asthma prevalence
    * :math:`p(\lambda)` is the prevalence of risk factor level :math:`\lambda`,
      ``risk_factor_prob[λ]``
    * :math:`\beta_{\lambda}` is the parameter for risk factor level :math:`\lambda`,
      ``asthma_prev_risk_factor_params[λ]``

    Args:
        asthma_prev_risk_factor_params: A vector of parameters for the risk factors, with
            shape ``(n - 1, 1)``.
        risk_factor_prob: A vector of the prevalence of the risk factor levels, with shape
            ``(n, 1)``.
    
    Returns:
        The correction term for asthma prevalence.
    """

    # prev_correction: correction term for prevalence
    prev_correction = np.dot(risk_factor_prob[1:], asthma_prev_risk_factor_params)
    return prev_correction


def compute_asthma_prevalence_λ(
    asthma_prev_risk_factor_params: list[float],
    odds_ratio_target: list[float],
    risk_factor_prob: list[float],
    β_0: float
) -> np.ndarray:
    r"""Compute the asthma prevalence based on the risk factors and the parameters provided.

    .. math::

        \zeta_{\lambda} = \sigma(\beta_0 + \log(\omega_{\lambda}) - \alpha)

    where:

    * :math:`\beta_0 = \sigma^{-1}(\eta)`
    * :math:`\omega_{\lambda}` is the odds ratio for risk factor level :math:`\lambda`,
      ``odds_ratio_target[λ]``
    * :math:`\alpha` is the correction term for the asthma prevalence, computed by
      ``get_asthma_prevalence_correction``

    Args:
        asthma_prev_risk_factor_params: A vector of parameters for the risk factors, with
            shape ``(n - 1, 1)``.
        odds_ratio_target: A vector of odds ratios between the risk factors and asthma, with
            shape ``(n, 1)``.
        risk_factor_prob: A vector of the prevalence of the risk factor levels, with shape
            ``(n, 1)``.
        β_0: The intercept of the logistic regression model.
    
    Returns:
        The calibrated asthma prevalence.
    """

    n = len(odds_ratio_target)

    # prev_correction: correction term for prevalence
    prev_correction = get_asthma_prevalence_correction(
        asthma_prev_risk_factor_params, risk_factor_prob
    )

    # asthma_prev_λ: asthma prevalence at risk factor level λ
    asthma_prev_λ = logistic.cdf(
        β_0 * np.ones(n) + 
        np.log(odds_ratio_target) - 
        prev_correction * np.ones(n)
    )

    return asthma_prev_λ


def compute_asthma_prevalence(
    asthma_prev_risk_factor_params: list[float],
    odds_ratio_target: list[float],
    risk_factor_prob: list[float],
    β_0: float
) -> float:
    r"""Compute the asthma prevalence based on the risk factors and the parameters provided.

    We want to find the calibrated asthma prevalence :math:`\zeta`:

    .. math::

        \zeta &= \sum_{\lambda=0}^{n} p(\lambda) \zeta_{\lambda} \\

    where:

    * :math:`p(\lambda)` is the probability of risk factor level :math:`\lambda`,
      ``risk_factor_prob[λ]``
    * :math:`\zeta_{\lambda}` is the predicted asthma prevalence at risk factor level
      :math:`\lambda`, ``asthma_prev_λ``

    We compute :math:`\zeta_{\lambda}` as follows:

    .. math::

        \zeta_{\lambda} &= \sigma(\beta_0 + \log(\omega_{\lambda}) - \alpha) \\
        \beta_0 &= \sigma^{-1}(\eta) \\
        \alpha &= \sum_{\lambda=1}^{n} p(\lambda) \cdot \beta_{\lambda}

    where:

    * :math:`\eta` is the target asthma prevalence, ``asthma_prev_target``, from the model of
      the BC Ministry of Health data.
    * :math:`\omega_{\lambda}` is the odds ratio for risk factor level :math:`\lambda`,
      ``odds_ratio_target[λ]``
    * :math:`\beta_{\lambda}` is the parameter for risk factor level :math:`\lambda`,
      ``asthma_prev_risk_factor_params[λ]``
    * :math:`\alpha` is the correction term for the asthma prevalence, computed by
      ``get_asthma_prevalence_correction``

    Args:
        asthma_prev_risk_factor_params: A vector of parameters for the risk factors, with
            shape ``(n - 1, 1)``.
        odds_ratio_target: A vector of odds ratios between the risk factors and asthma, with
            shape ``(n, 1)``.
        risk_factor_prob: A vector of the prevalence of the risk factor levels, with shape
            ``(n, 1)``.
        β_0: The intercept of the logistic regression model.
    
    Returns:
        The calibrated asthma prevalence.
    """

    # asthma_prev_λ: asthma prevalence at risk factor level λ
    asthma_prev_λ = compute_asthma_prevalence_λ(
        asthma_prev_risk_factor_params, odds_ratio_target, risk_factor_prob, β_0
    )

    # asthma_prev: predicted asthma prevalence
    asthma_prev = np.dot(asthma_prev_λ, risk_factor_prob)
    return(asthma_prev)



def compute_asthma_prevalence_difference(
    asthma_prev_risk_factor_params: list[float],
    odds_ratio_target: list[float],
    risk_factor_prob: list[float],
    β_0: float,
    asthma_prev_target: float
) -> float:
    r"""Compute the absolute difference between the calibrated and target asthma prevalence.

    We want to find:

    .. math::

        |\zeta - \eta|

    where:

    * :math:`\zeta` is the calibrated asthma prevalence, computed by
      ``compute_asthma_prevalence``
    * :math:`\eta` is the target asthma prevalence, ``asthma_prev_target``, from the model of
      the BC Ministry of Health data.

    Args:
        asthma_prev_risk_factor_params: A vector of parameters for the risk factors, with
            shape ``(n - 1, 1)``.
        odds_ratio_target: A vector of odds ratios between the risk factors and asthma, with
            shape ``(n, 1)``.
        risk_factor_prob: A vector of the prevalence of the risk factor levels, with shape
            ``(n, 1)``.
        β_0: The intercept of the logistic regression model.
        asthma_prev_target: The target prevalence of asthma.

    Returns:
        The absolute difference between the calibrated and target asthma prevalence.
    """

    asthma_prev_calibrated = compute_asthma_prevalence(
        asthma_prev_risk_factor_params, odds_ratio_target, risk_factor_prob, β_0
    )
    return np.abs(asthma_prev_calibrated - asthma_prev_target)



def optimize_prevalence_β_parameters(
    asthma_prev_target: float,
    odds_ratio_target: list[float],
    risk_factor_prob: list[float],
    β_0: float | None = None,
    verbose: bool = False
) -> list[float]:
    r"""Calibrate asthma prevalence based on the target prevalence and odds ratios of risk factors.

    We want to find the parameters :math:`\beta_{\lambda}` such that the difference between the
    calibrated asthma prevalence and the target asthma prevalence is minimized. The calibrated
    asthma prevalence is computed as follows:

    .. math::

        \beta_0 &= \sigma^{-1}(\eta) \\
        \zeta_{\lambda} &= \sigma(\beta_0 + \log(\omega_{\lambda}) - \alpha) \\
        \alpha &= \sum_{\lambda=1}^{n} p(\lambda) \cdot \beta_{\lambda} \\
        \zeta &= \sum_{\lambda=0}^{n} p(\lambda) \zeta_{\lambda}

    where:

    * :math:`\eta` is the target asthma prevalence, ``asthma_prev_target``, from the model of
      the BC Ministry of Health data.
    * :math:`\omega_{\lambda}` is the odds ratio for risk factor level :math:`\lambda`,
      ``odds_ratio_target[i]``
    * :math:`p(\lambda)` is the prevalence of risk factor level :math:`\lambda`,
      ``risk_factor_prob[i]``
    * :math:`\beta_{\lambda}` is the parameter for risk factor level :math:`\lambda`,
      ``asthma_prev_risk_factor_params[i]``
    * :math:`\alpha` is the correction term for the asthma prevalence
    * :math:`\zeta_{\lambda}` is the predicted asthma prevalence at risk factor level :math:`\lambda`
    * :math:`\zeta` is the predicted / calibrated asthma prevalence

    The function uses the ``BFGS`` optimization algorithm to minimize the absolute difference
    between the calibrated asthma prevalence and the target asthma prevalence.

    Args:
        asthma_prev_target: The target prevalence of asthma from the BC Ministry of Health model.
        odds_ratio_target: A vector of odds ratios for the risk factors, with shape ``(n, 1)``.
        risk_factor_prob: A vector of the prevalence of the risk factors, with shape ``(n, 1)``.
        β_0: The intercept of the logistic regression model. If ``None``, it is set to
            the ``logit`` of the target prevalence.
        verbose: A boolean indicating if the trace should be printed.

    Returns:
        A vector of the asthma prevalence beta parameters for each risk factor level, with shape
        ``(n - 1, 1)``.
    """

    if β_0 is None:
        β_0 = logit(asthma_prev_target)

    # set initial beta parameters to 0
    asthma_prev_risk_factor_params = np.zeros(len(odds_ratio_target) - 1)

    res = optimize.minimize(
        fun=compute_asthma_prevalence_difference,
        x0=asthma_prev_risk_factor_params,
        args=(odds_ratio_target, risk_factor_prob, β_0, asthma_prev_target),
        method="BFGS",
        tol=1e-15,
        hess=True,
        options={"maxiter": 10000, "disp": verbose}
    )

    # return optimized beta parameters
    asthma_prev_risk_factor_params = res.x
    return asthma_prev_risk_factor_params






