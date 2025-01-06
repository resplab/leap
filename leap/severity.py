from __future__ import annotations
import numpy as np
from scipy.special import gamma
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.agent import Agent
    from leap.control import Control
    from leap.exacerbation import Exacerbation

logger = get_logger(__name__)


class SeverityLevels:
    """A class containing the probability of each asthma exacerbation severity level.

    Attributes:
        mild: The probability of an asthma exacerbation being mild.
        moderate: The probability of an asthma exacerbation being moderate.
        severe: The probability of an asthma exacerbation being severe.
        very_severe: The probability of an asthma exacerbation being very severe
    """
    def __init__(
        self,
        mild: float,
        moderate: float,
        severe: float,
        very_severe: float
    ):
        self.mild = mild
        self.moderate = moderate
        self.severe = severe
        self.very_severe = very_severe

    def as_array(self) -> np.ndarray:
        """Return the severity levels as an array."""
        return np.array([self.mild, self.moderate, self.severe, self.very_severe])


class ExacerbationSeverity:
    """A class containing information about asthma exacerbation severity.

    There are four levels of asthma exacerbation severity:

    * 1 = mild
    * 2 = moderate
    * 3 = severe
    * 4 = very severe

    """
    def __init__(
        self,
        config: dict | None = None,
        hyperparameters: dict | None = None,
        parameters: dict | None = None
    ):
        if config is not None:
            self.hyperparameters = config["hyperparameters"]
            self.parameters = config["parameters"]
        elif hyperparameters is not None and parameters is not None:
            self.hyperparameters = hyperparameters
            self.parameters = parameters
        else:
            raise ValueError(
                "Either config dict or hyperparameters and parameters must be provided."
            )

        self.assign_random_p()

    @property
    def hyperparameters(self) -> dict:
        """A dictionary containing the hyperparameters used in the Dirichlet-multinomial distribution.
        
        See:
        `Numpy Dirichlet function
        <https://numpy.org/doc/2.1/reference/random/generated/numpy.random.dirichlet.html>`_.

        A dictionary containing the following keys:

        * ``k``: integer, number of trials.
        * ``α``: parameter vector, length 4.
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters: dict):
        KEYS = ["k", "α"]
        for key in KEYS:
            if key not in hyperparameters:
                raise ValueError(f"The key '{key}' is missing in the hyperparameters.")
        if len(hyperparameters["α"]) != 4:
            raise ValueError("The length of the 'α' vector must be 4.")
        self._hyperparameters = hyperparameters

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:

        * ``βprev_hosp_ped``: ``float``, parameter for previous hospitalizations due to asthma
          in childhood.
        * ``βprev_hosp_adult``: ``float``, parameter for previous hospitalizations due to asthma
          in adulthood.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["βprev_hosp_ped", "βprev_hosp_adult"]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"The key '{key}' is missing in the parameters.")
        self._parameters = parameters

    @property
    def severity_levels(self) -> SeverityLevels:
        """The probability of each asthma exacerbation severity level.
        
        A probability vector giving the probability of each exacerbation type,
        computed using the Dirichlet-multinomial distribution.
        """
        return self._severity_levels
    
    @severity_levels.setter
    def severity_levels(self, severity_levels: SeverityLevels):
        self._severity_levels = severity_levels

    def assign_random_p(self):
        """Compute the probability vector ``p`` from the Dirichlet distribution.

        See:
        `Numpy Dirichlet function
        <https://numpy.org/doc/2.1/reference/random/generated/numpy.random.dirichlet.html>`_.
        """

        p = np.random.dirichlet(np.array(self.hyperparameters["α"]) * self.hyperparameters["k"])
        self.severity_levels = SeverityLevels(p[0], p[1], p[2], p[3])

    def compute_distribution(self, num_current_year: int, prev_hosp: bool, age: int) -> np.ndarray:
        """Compute the exacerbation severity distribution.

        Compute the exacerbation severity distribution for a patient in a given year using the
        Dirichlet probability vector ``p`` in the Multinomial distribution. See:
        https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.Multinomial.

        For example, if the patient has ``num_current_year = 10`` exacerbations in the current year,
        then the output might be:

        .. code-block::

            mild | moderate | severe | very severe
            2    | 1        | 6      | 1

        Args:
            num_current_year: the number of asthma exacerbations the patient has had this year.
                Will be used as the number of trials in the Multinomial distribution.
            prev_hosp: has patient been previously hospitalized for asthma?
            age: the age of the person in years.

        Returns:
            The distribution of asthma exacerbations by exacerbation type for the current year.

        Examples:

            >>> from leap.severity import ExacerbationSeverity
            >>> exacerbation_severity = ExacerbationSeverity(
            ...     hyperparameters={"α": [1000, 0.00001, 0.00001, 0.00001], "k": 100},
            ...     parameters={"βprev_hosp_ped": 1.79, "βprev_hosp_adult": 2.88}
            ... )
            >>> exacerbation_severity.compute_distribution(
            ...     num_current_year=10,
            ...     prev_hosp=True,
            ...     age=85
            ... ) # doctest: +NORMALIZE_WHITESPACE
            array([10, 0, 0, 0])
        """
        severity_levels = self.severity_levels
        severity_levels_array = severity_levels.as_array()

        if num_current_year == 0:
            return np.zeros(4)
        else:
            if prev_hosp:
                weight = severity_levels_array / np.sum(severity_levels_array)
                severity_levels.very_severe = severity_levels.very_severe * (
                    self.parameters["βprev_hosp_ped"] if age < 14
                    else self.parameters["βprev_hosp_adult"]
                )
                severity_levels_array = weight * (1 - severity_levels.very_severe)

            return np.random.multinomial(num_current_year, severity_levels_array)

    def compute_hospitalization_prob(
        self, agent: Agent, control: Control, exacerbation: Exacerbation
    ) -> bool:
        """Determine whether a person has been hospitalized due to an asthma exacerbation.

        https://stats.stackexchange.com/questions/174952/marginal-probability-function-of-the-dirichlet-multinomial-distribution

        .. note::

            Note on limits: the ``Γ(z)`` function approaches infinity as ``z -> 0+`` and ``z -> inf``.
            Empirically, when the ``total_rate`` variable is around ``150``, the ``Γ(z)`` function
            returns ``Inf``. Likewise, if the probability of a severe exacerbation is exactly ``1.0``,
            the ``Γ(z)`` function will return ``Inf``. To remedy this, I have added max values for
            these variables.

        Args:
            agent: A person in the simulation.
            control: Asthma control module.
            exacerbation: Asthma exacerbation module.

        Returns:
            The binary probability of a hospitalization.

        """
        max_age = agent.age - 2
        sex = agent.sex

        if max_age < 3:
            return False
        else:
            if agent.asthma_age is None:
                raise ValueError("Asthma age is not set.")

            year = agent.year - (agent.age - agent.asthma_age)
            total_rate = 0
            for age in range(agent.asthma_age, max_age + 1):
                control_levels = control.compute_control_levels(sex=sex, age=age)
                total_rate += exacerbation.compute_num_exacerbations(
                    age=age, sex=sex, year=year, control_levels=control_levels
                )
                year += 1

            # toss a coin: avg chance of having at least one hospitalization
            prob_severe_exacerbation = min(self.severity_levels.very_severe, 0.9999999999999)
            total_rate = min(total_rate, 150)
            zero_prob = (
                1 / gamma(total_rate + 1) *
                (gamma(total_rate + 1 - prob_severe_exacerbation) /
                 gamma(1 - prob_severe_exacerbation))
            )
            p = 1 - min(max(zero_prob, 0), 1)
            has_hospitalization = bool(np.random.binomial(1, p))
            return has_hospitalization


class ExacerbationSeverityHistory:
    """A class containing information about the history of asthma exacerbations by severity.

    There are four levels of asthma exacerbation severity:

    * 1 = mild
    * 2 = moderate
    * 3 = severe
    * 4 = very severe

    Attributes:
        current_year: An array of 4 integers indicating the number of exacerbations for
            that severity level in the current year.
        prev_year: An array of 4 integers indicating the number of exacerbations for
            that severity level in the previous year.
    """
    def __init__(self, current_year: np.ndarray, prev_year: np.ndarray):
        self.current_year = current_year
        self.prev_year = prev_year
