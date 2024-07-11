import pathlib
import pandas as pd
import numpy as np
from leap.utils import PROCESSED_DATA_PATH, sigmoid, logit
from leap.logger import get_logger

logger = get_logger(__name__)


class Incidence:
    """A class containing information about asthma incidence.

    Attributes:
        hyperparameters (dict): A dictionary containing the hyperparameters used to compute
            ``β0`` from a normal distribution:
            * ``β0_μ``: float, the mean of the normal distribution.
            * ``β0_σ``: float, the standard deviation of the normal distribution.
        parameters (dict): A dictionary containing the following keys:
            * ``β0``: float, a constant parameter, randomly selected from a normal distribution
              with mean ``β0_μ`` and standard deviation ``β0_σ``. See ``hyperparameters``.
            * ``βsex``: float, the parameter for the sex term, i.e. ``βsex * sex``.
            * ``βage``: list[float], an array of 5 parameters to be multiplied by functions of age,
              i.e.

              .. code-block:: python

                βage1 * f1(age) + βage2 * f2(age) + βage3 * f3(age) +
                βage4 * f4(age) + βage5 * f5(age)

              See ``poly_age_calculator``.
            * ``βyear``: float, the parameter for the year term, i.e. ``βyear * year``.
            * ``βsexage``: list[float], an array of 5 parameters to be multiplied by the sex and
              functions of age, i.e.

              .. code-block:: python

                βsexage1 * f1(age) * sex + βsexage2 * f2(age) * sex + βsexage3 * f3(age) * sex +
                βsexage4 * f4(age) * sex + βsexage5 * f5(age) * sex

              See ``poly_age_calculator``.
            * ``βsexyear``: float, the parameter to be multiplied by sex and year,
              i.e. ``βsexyear * year * sex``.
            * ``βfam_hist``: list[float], an array of 2 parameters to be multiplied by functions of
              age. See ``log_OR_family_history``.
            * ``βabx_exp``: list[float], an array of 3 parameters to be multiplied by functions of
              age and antibiotic exposure. See ``log_OR_abx_exposure``.
        min_year (int): TODO.
        max_year (int): TODO.
        max_age (int): TODO.
        correction_table (pd.api.typing.DataFrameGroupBy): A dataframe grouped by year, age, and sex.
            Each dataframe contains the following columns:
                * ``year``: integer year.
                * ``sex``: 0 = female, 1 = male.
                * ``age``: integer age.
                * ``correction``: float, TODO.
            See ``master_occurrence_correction.csv``.
    """
    def __init__(
        self,
        config: dict | None = None,
        hyperparameters: dict | None = None,
        parameters: dict | None = None,
        max_age: int = 110,
        correction_table: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if config is None and (parameters is None or hyperparameters is None):
            raise ValueError("Either config dict or parameters must be provided.")
        elif config is not None:
            self.hyperparameters = config["hyperparameters"]
            self.parameters = config["parameters"]
            self.parameters["βage"] = np.array(self.parameters["βage"])
            self.parameters["βsexage"] = np.array(self.parameters["βsexage"])
            self.parameters["βfam_hist"] = np.array(self.parameters["βfam_hist"])
            self.parameters["βabx_exp"] = np.array(self.parameters["βabx_exp"])
            self.max_age = config["max_age"]
        else:
            self.hyperparameters = hyperparameters
            self.parameters = parameters
            self.max_age = max_age

        if correction_table is None:
            self.correction_table = self.load_incidence_correction_table()
        else:
            self.correction_table = correction_table

        years = np.unique([key[0] for key in self.correction_table.groups.keys()])
        self.min_year = int(np.min(years)) + 1
        self.max_year = int(np.max(years))

    def load_incidence_correction_table(self):
        """Load the asthma incidence correction table.

        Returns:
            pd.api.typing.DataFrameGroupBy: A dataframe grouped by year, age, and sex.
            Each dataframe contains the following columns:
                * ``year``: integer year.
                * ``sex``: 0 = female, 1 = male.
                * ``age``: integer age.
                * ``correction``: float, TODO.
        """
        df = pd.read_csv(
            pathlib.Path(PROCESSED_DATA_PATH, "master_asthma_occurrence_correction.csv")
        )
        df = df[df["type"] == "inc"]
        df.drop(columns=["type"], inplace=True)
        grouped_df = df.groupby(["year", "sex", "age"])
        return grouped_df

    def equation(self, sex: bool, age: int, year: int, has_family_hist: bool, dose: int):
        """Compute the asthma incidence equation.

        Args:
            sex (int): 0 = female, 1 = male.
            age (int): The age of the agent.
            year (int): The calendar year.
            has_family_hist (bool): Whether the agent has a family history of asthma.
            dose (int): TODO.
        """

        correction_year = min(year, self.max_year + 1)
        year = min(year, self.max_year)
        p0 = self.crude_incidence(sex, age, year)
        p = sigmoid(
            logit(p0) +
            has_family_hist * self.log_OR_family_history(age) +
            self.log_OR_abx_exposure(age, dose) +
            self.correction_table.get_group(
                (correction_year, sex, min(age, 63))
            )["correction"].values[0]
        )
        return p

    def crude_incidence(self, sex, age, year: int):
        poly_age = self.poly_age_calculator(age)
        return np.exp(
            self.parameters["β0"] +
            self.parameters["βsex"] * sex +
            self.parameters["βyear"] * year +
            self.parameters["βsexyear"] * sex * year +
            np.dot(self.parameters["βage"], poly_age) +
            np.dot(self.parameters["βsexage"], poly_age) * sex
        )

    def log_OR_family_history(self, age: int):
        return self.parameters["βfam_hist"][0] + (min(5, age) - 3) * self.parameters["βfam_hist"][1]

    def log_OR_abx_exposure(self, age: int, dose: int):
        if age > 7 or dose == 0:
            return 0
        else:
            return (
                self.parameters["βabx_exp"][0] +
                self.parameters["βabx_exp"][1] * min(age, 7) +
                self.parameters["βabx_exp"][2] * min(dose, 3)
            )

    def poly_age_calculator(
        self,
        age: int,
        alpha: list[float] = [32.07692, 32.42755, 32.76123, 32.80415, 32.54075],
        nd: list[float] = [
            1, 520, 179636.923076923, 47536813.3328764, 11589923664.2537,
            2683688761696.54, 594554071731935
        ]
    ):
        fs = np.zeros(6)
        fs[0] = 1 / np.sqrt(nd[1])
        fs[1] = (age - alpha[0]) / np.sqrt(nd[2])

        for i in range(1, 5):
            fs[i + 1] = (
                (age - alpha[i]) * np.sqrt(nd[i + 1]) * fs[i] -
                nd[i + 1] / np.sqrt(nd[i]) * fs[i - 1]
            ) / np.sqrt(nd[i + 2])
        return fs[1:]


def agent_has_asthma(
    agent, incidence: Incidence, prevalence, age: int | None = None, year: int | None = None
) -> bool:
    """Determine whether the agent obtains a new asthma diagnosis based on age and sex.

    Args:
        agent (Agent): An agent in the model.
        incidence (Incidence): Asthma incidence.
        prevalence: TODO.
        age (int): The age of the agent.
        year (int): The calendar year.
    """
    if age is None:
        age = min(agent.age, incidence.max_age)
    if year is None:
        year = agent.year

    if age < 3:
        has_asthma = False
    elif age == 3:
        has_asthma = bool(np.random.binomial(1, prevalence.equation(
            agent.sex, age, year, agent.has_family_hist, agent.num_antibiotic_use
        )))
    else:
        has_asthma = bool(np.random.binomial(1, incidence.equation(
            agent.sex, age, year, agent.has_family_hist, agent.num_antibiotic_use
        )))
    return has_asthma


def compute_asthma_age(
    agent, incidence: Incidence, prevalence, current_age: int, max_asthma_age: int = 110
) -> int:
    """Compute the age at which the person (agent) is first diagnosed with asthma.

    Args:
        agent (Agent): A person in the model.
        incidence (Incidence): Asthma incidence.
        prevalence (Prevalence): Asthma prevalence.
        current_age (int): The current age of the agent.
    """
    # obtain the previous incidence
    min_year = incidence.min_year
    max_year = incidence.max_year

    if current_age == 3:
        return 3
    else:
        find_asthma_age = True
        asthma_age = 3
        year = min(agent.year - current_age + asthma_age, min_year)
        while find_asthma_age and asthma_age < max_asthma_age:
            has_asthma = agent_has_asthma(agent, incidence, prevalence, asthma_age, year)
            if has_asthma:
                return asthma_age
            asthma_age += 1
            asthma_age = min(asthma_age, incidence.max_age)
            year += 1
            year = min(year, max_year)
        return asthma_age
