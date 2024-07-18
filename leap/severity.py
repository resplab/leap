import numpy as np
from scipy.special import gamma


class ExacerbationSeverity:
    """A class containing information about asthma exacerbation severity.

    There are four levels of asthma exacerbation severity:
        1 = mild
        2 = moderate
        3 = severe
        4 = very severe

    Attributes:
        hyperparameters (dict): A dictionary containing the hyperparameters used
            in the Dirichlet-multinomial distribution. See
            https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.Dirichlet.
            * ``k``: integer, number of trials.
            * ``α``: parameter vector, length 4.
        parameters (dict): A dictionary containing the following keys:
            * ``p``: a probability vector giving the probability of each exacerbation type,
              using the Dirichlet-multinomial distribution.
            * ``βprev_hosp_ped``: Float64, parameter for previous hospitalizations due to asthma
              in childhood.
            * ``βprev_hosp_adult``: Float64, parameter for previous hospitalizations due to asthma
              in adulthood.
    """
    def __init__(
        self,
        config: dict | None = None,
        hyperparameters: dict | None = None,
        parameters: dict | None = None
    ):
        if config is None and hyperparameters is None and parameters is None:
            raise ValueError(
                "Either config dict or hyperparameters and parameters must be provided."
            )
        elif config is not None:
            self.hyperparameters = config["hyperparameters"]
            self.parameters = config["parameters"]
            self.parameters["p"] = self.assign_random_p()
        else:
            self.hyperparameters = hyperparameters
            self.parameters = parameters

    def assign_random_p(self):
        """Compute the probability vector ``p`` from the Dirichlet distribution.

        Compute the probability vector `p` from the Dirichlet distribution. See:
        https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.Dirichlet.

        Returns
            np.ndarray: the probability vector `p`.
        """

        p = np.random.dirichlet(self.hyperparameters["α"] * self.hyperparameters["k"])
        self.parameters["p"] = p

    def compute_distribution(self, num_current_year, prev_hosp, age):
        """Compute the exacerbation severity distribution.

        Compute the exacerbation severity distribution for a patient in a given year using the
        Dirichlet probability vector `p` in the Multinomial distribution. See:
        https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.Multinomial.

        For example, if the patient has ``num_current_year = 10`` exacerbations in the current year,
        then the output might be:

        mild | moderate | severe | very severe
        2    | 1        | 6      | 1

        Args:
            num_current_year (int): the number of asthma exacerbations the patient has had this year.
                Will be used as the number of trials in the Multinomial distribution.
            prev_hosp (bool): has patient been previously hospitalized for asthma?
            age (int): the age of the person in years.

        Returns
            np.ndarray: the distribution of asthma exacerbations by exacerbation type for the
            current year.
        """
        p = self.parameters["p"]
        index_very_severe = 4
        index_max = index_very_severe

        if num_current_year == 0:
            return np.zeros(index_max)
        else:
            if prev_hosp:
                weight = p[0:3]
                weight = weight / np.sum(weight)
                p[index_very_severe] = p[index_very_severe] * (
                    self.parameters["βprev_hosp_ped"] if age < 14
                    else self.parameters["βprev_hosp_adult"]
                )
                p[0:3] = weight * (1 - p[index_very_severe])

            return np.random.multinomial(num_current_year, p)

    def compute_hospitalization_prob(self, agent, control, exacerbation) -> bool:
        """Determine whether a person has been hospitalized due to an asthma exacerbation.

        https://stats.stackexchange.com/questions/174952/marginal-probability-function-of-the-dirichlet-multinomial-distribution

        Note on limits: the Γ(z) function approaches infinity as z -> 0+ and z -> inf.
        Empirically, when the ``total_rate`` variable is around 150, the Γ(z) function returns Inf.
        Likewise, if the probability of a severe exacerbation is exactly 1.0, the Γ(z) function
        will return Inf. To remedy this, I have added max values for these variables.

        Args:
            agent (Agent): a person in the simulation.
            control (Control): asthma control module.
            exacerbation (Exacerbation): asthma exacerbation module.

        Returns:
            bool: the binary probability of a hospitalization.

        """
        max_age = agent.age - 2
        sex = agent.sex

        if max_age < 3:
            return 0
        else:
            year = agent.year - (agent.age - agent.asthma_age)
            total_rate = 0
            for age in range(agent.asthma_age, max_age + 1):
                control_levels = control.compute_control_levels(sex=sex, age=age)
                total_rate += exacerbation.compute_num_exacerbations(
                    age=age, sex=sex, year=year, control_levels=control_levels
                )
                year += 1

            # toss a coin: avg chance of having at least one hosp
            prob_severe_exacerbation = min(self.parameters["p"][3], 0.9999999999999)
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
        1 = mild
        2 = moderate
        3 = severe
        4 = very severe

    Attributes:
        current_year (np.ndarray): An array of 4 integers indicating the number of exacerbations for
            that severity level in the current year.
        prev_year (np.ndarray): An array of 4 integers indicating the number of exacerbations for
            that severity level in the previous year.
    """
    def __init__(self, current_year: np.ndarray, prev_year: np.ndarray):
        self.current_year = current_year
        self.prev_year = prev_year
