from __future__ import annotations
import abc
import pandas as pd
import numpy as np
from leap.utils import get_data_path, sigmoid, logit
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.agent import Agent

logger = get_logger(__name__)


class Occurrence:
    """A class containing information about asthma occurrence (incidence or prevalence)."""

    def __init__(
        self,
        config: dict | None = None,
        hyperparameters: dict | None = None,
        parameters: dict | None = None,
        max_age: int = 110,
        correction_table: DataFrameGroupBy | None = None
    ):
        if config is not None:
            self.hyperparameters = config["hyperparameters"]
            self.parameters = config["parameters"]
            self.max_age = config["max_age"]
        elif hyperparameters is not None and parameters is not None:
            self.hyperparameters = hyperparameters
            self.parameters = parameters
            self.max_age = max_age
        else:
            raise ValueError(
                "Either config dict or parameters and hyperparameters must be provided."
            )

        if correction_table is None:
            self.correction_table = self.load_occurrence_correction_table()
        else:
            self.correction_table = correction_table

        years = np.unique([key[0] for key in self.correction_table.groups.keys()])
        self.min_year = int(np.min(years)) + 1
        self.max_year = int(np.max(years))

    @property
    def hyperparameters(self) -> dict:
        """A dictionary containing the hyperparameters used to compute ``β0`` from a normal
        distribution:
        
        * ``β0_μ``: float, the mean of the normal distribution.
        * ``β0_σ``: float, the standard deviation of the normal distribution.

        """
        return self._hyperparameters
    
    @hyperparameters.setter
    def hyperparameters(self, hyperparameters: dict):
        KEYS = ["β0_μ", "β0_σ"]
        for key in KEYS:
            if key not in hyperparameters.keys():
                raise ValueError(f"Missing key {key} in hyperparameters.")
        self._hyperparameters = hyperparameters

    @property
    def parameters(self) -> dict:
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        self._parameters = parameters

    @property
    def correction_table(self) -> DataFrameGroupBy:
        """A dataframe grouped by year, age, and sex.
        
        Each dataframe contains the following columns:

        * ``year``: integer year.
        * ``sex``: 0 = female, 1 = male.
        * ``age``: integer age.
        * ``correction``: float, TODO.

        See ``master_occurrence_correction.csv``.
        """
        return self._correction_table
    
    @correction_table.setter
    def correction_table(self, correction_table: DataFrameGroupBy):
        self._correction_table = correction_table

    @property
    def max_age(self) -> int:
        """The maximum age of agents in the model."""
        return self._max_age
    
    @max_age.setter
    def max_age(self, max_age: int):
        self._max_age = max_age

    @property
    def min_year(self) -> int:
        return self._min_year
    
    @min_year.setter
    def min_year(self, min_year: int):
        self._min_year = min_year

    @property
    def max_year(self) -> int:
        return self._max_year
    
    @max_year.setter
    def max_year(self, max_year: int):
        self._max_year = max_year

    def load_occurrence_correction_table(
        self, occurrence_type: str
    ) -> DataFrameGroupBy:
        """Load the asthma incidence correction table.

        Returns:
            Dataframe grouped by year, age, and sex. Each dataframe contains the following columns:

            * ``year``: integer year.
            * ``sex``: 0 = female, 1 = male.
            * ``age``: integer age.
            * ``correction``: float, TODO.

        """
        df = pd.read_csv(
            get_data_path("processed_data", "master_asthma_occurrence_correction.csv")
        )
        df = df[df["type"] == occurrence_type]
        df.drop(columns=["type"], inplace=True)
        grouped_df = df.groupby(["year", "sex", "age"])
        return grouped_df

    def equation(
        self, sex: int, age: int, year: int, has_family_history: bool, dose: int
    ) -> float:
        """Compute the asthma occurrence equation.

        Args:
            sex: 0 = female, 1 = male.
            age: The age of the agent.
            year: The calendar year.
            has_family_history: Whether the agent has a family history of asthma.
            dose: TODO.
        """
        correction_year = min(year, self.max_year)
        year = min(year, self.max_year)
        p0 = self.crude_occurrence(sex, age, year)
        p = sigmoid(
            logit(p0) +
            has_family_history * self.log_OR_family_history(age) +
            self.log_OR_abx_exposure(age, dose) +
            self.correction_table.get_group(
                (correction_year, sex, min(age, 63))
            )["correction"].values[0]
        )
        return p

    @abc.abstractmethod
    def crude_occurrence(self, sex, age: int, year: int) -> float:
        return

    def log_OR_family_history(self, age: int) -> float:
        return self.parameters["βfam_hist"][0] + (min(5, age) - 3) * self.parameters["βfam_hist"][1]

    def log_OR_abx_exposure(self, age: int, dose: int) -> float:
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
    ) -> np.ndarray:
        fs = np.zeros(6)
        fs[0] = 1 / np.sqrt(nd[1])
        fs[1] = (age - alpha[0]) / np.sqrt(nd[2])

        for i in range(1, 5):
            fs[i + 1] = (
                (age - alpha[i]) * np.sqrt(nd[i + 1]) * fs[i] -
                nd[i + 1] / np.sqrt(nd[i]) * fs[i - 1]
            ) / np.sqrt(nd[i + 2])
        return fs[1:]

    def poly_year_calculator(
        self,
        year: int,
        alpha: list[float] = [2009.5, 2009.5],
        nd: list[float] = [1.0, 520.0, 17290.0, 456456.0]
    ) -> np.ndarray:
        fs = np.zeros(3)
        fs[0] = 1 / np.sqrt(nd[1])
        fs[1] = (year - alpha[0]) / np.sqrt(nd[2])
        fs[2] = ((year - alpha[1]) * np.sqrt(nd[2]) * fs[1] - nd[2] / np.sqrt(nd[1]) * fs[0]) / np.sqrt(nd[3])
        return fs[1:]


class Incidence(Occurrence):
    """A class containing information about asthma incidence."""

    def __init__(
        self,
        config: dict | None = None,
        hyperparameters: dict | None = None,
        parameters: dict | None = None,
        max_age: int = 110,
        correction_table: DataFrameGroupBy | None = None
    ):
        super().__init__(config, hyperparameters, parameters, max_age, correction_table)
        self.parameters["βage"] = np.array(self.parameters["βage"])
        self.parameters["βsexage"] = np.array(self.parameters["βsexage"])
        self.parameters["βfam_hist"] = np.array(self.parameters["βfam_hist"])
        self.parameters["βabx_exp"] = np.array(self.parameters["βabx_exp"])

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:
        
            * ``β0 (float)``: A constant parameter, randomly selected from a normal distribution
              with mean ``β0_μ`` and standard deviation ``β0_σ``. See ``hyperparameters``.
            * ``βsex (float)``: The parameter for the sex term, i.e. ``βsex * sex``.
            * ``βage (list[float])``: An array of 5 parameters to be multiplied by functions of age,
              i.e.

              .. code-block:: python

                βage1 * f1(age) + βage2 * f2(age) + βage3 * f3(age) +
                βage4 * f4(age) + βage5 * f5(age)

              See ``poly_age_calculator``.
            * ``βyear (float)``: The parameter for the year term, i.e. ``βyear * year``.
            * ``βsexage (list[float])``: An array of 5 parameters to be multiplied by the sex and
              functions of age, i.e.

              .. code-block:: python

                βsexage1 * f1(age) * sex + βsexage2 * f2(age) * sex + βsexage3 * f3(age) * sex +
                βsexage4 * f4(age) * sex + βsexage5 * f5(age) * sex

              See ``poly_age_calculator``.
            * ``βsexyear (float)``: The parameter to be multiplied by sex and year,
              i.e. ``βsexyear * year * sex``.
            * ``βfam_hist (list[float])``: An array of 2 parameters to be multiplied by functions of
              age. See ``log_OR_family_history``.
            * ``βabx_exp (list[float])``: An array of 3 parameters to be multiplied by functions of
              age and antibiotic exposure. See ``log_OR_abx_exposure``.

        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["β0", "βsex", "βage", "βyear", "βsexage", "βsexyear", "βfam_hist", "βabx_exp"]
        for key in KEYS:
            if key not in parameters.keys():
                raise ValueError(f"Missing key {key} in parameters.")
        self._parameters = parameters

    def load_occurrence_correction_table(self) -> DataFrameGroupBy:
        """Load the asthma incidence correction table.

        Returns:
            A dataframe grouped by year, age, and sex.
            Each dataframe contains the following columns:

            * ``year``: integer year.
            * ``sex``: 0 = female, 1 = male.
            * ``age``: integer age.
            * ``correction``: float, TODO.

        """
        grouped_df = super().load_occurrence_correction_table(occurrence_type="inc")
        return grouped_df

    def crude_occurrence(self, sex, age: int, year: int) -> float:
        poly_age = self.poly_age_calculator(age)
        return np.exp(
            self.parameters["β0"] +
            self.parameters["βsex"] * sex +
            self.parameters["βyear"] * year +
            self.parameters["βsexyear"] * sex * year +
            np.dot(self.parameters["βage"], poly_age) +
            np.dot(self.parameters["βsexage"], poly_age) * sex
        )


class Prevalence(Occurrence):
    """A class containing information about asthma prevalence."""

    def __init__(
        self,
        config: dict | None = None,
        hyperparameters: dict | None = None,
        parameters: dict | None = None,
        max_age: int = 110,
        correction_table: DataFrameGroupBy | None = None
    ):
        super().__init__(config, hyperparameters, parameters, max_age, correction_table)
        self.parameters["βage"] = np.array(self.parameters["βage"])
        self.parameters["βsexage"] = np.array(self.parameters["βsexage"])
        self.parameters["βsexyear"] = np.array(self.parameters["βsexyear"])
        self.parameters["βyearage"] = np.array(self.parameters["βyearage"])
        self.parameters["βsexyearage"] = np.array(self.parameters["βsexyearage"])
        self.parameters["βfam_hist"] = np.array(self.parameters["βfam_hist"])
        self.parameters["βabx_exp"] = np.array(self.parameters["βabx_exp"])

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:

            * ``β0 (float)``: A constant parameter, randomly selected from a normal distribution
              with mean ``β0_μ`` and standard deviation ``β0_σ``. See ``hyperparameters``.
            * ``βsex (float)``: The parameter for the sex term, i.e. ``βsex * sex``.
            * ``βage (list[float])``: An array of 5 parameters to be multiplied by functions of age,
              i.e.

              .. code-block:: python

                βage1 * f1(age) + βage2 * f2(age) + βage3 * f3(age) +
                βage4 * f4(age) + βage5 * f5(age)

              See ``poly_age_calculator``.

            * ``βyear (list[float])``: An array of 2 parameters to be multiplied by functions of
              year, i.e. ``βyear1 * g1(year) + βyear2 * g2(year)``. See ``poly_year_calculator``.
            * ``βsexage (list[float])``: An array of 5 parameters to be multiplied by the sex and
              functions of age, i.e.

              .. code-block:: python

                βsexage1 * f1(age) * sex + βsexage2 * f2(age) * sex + βsexage3 * f3(age) * sex +
                βsexage4 * f4(age) * sex + βsexage5 * f5(age) * sex

              See ``poly_age_calculator``.
            * ``βsexyear (list[float])``: An array of 2 parameters to be multiplied by sex and
              functions of year, i.e. ``βyear1 * g1(year) + βyear2 * g2(year)``.
            * ``βyearage (list[float])``: An array of 10 parameters to be multiplied by functions of
              age and year, i.e.

              .. code-block:: python

                βyearage1 * f1(age) * g1(year) + βyearage2 * f1(age) * g2(year) +
                βyearage3 * f2(age) * g1(year) + βyearage4 * f2(age) * g2(year) +
                βyearage5 * f3(age) * g1(year) + βyearage6 * f3(age) * g2(year) +
                βyearage7 * f4(age) * g1(year) + βyearage8 * f4(age) * g2(year) +
                βyearage9 * f5(age) * g1(year) + βyearage10 * f5(age) * g2(year)

              See ``poly_age_calculator`` and ``poly_year_calculator``.
            * ``βsexyearage (list[float])``: An array of 10 parameters to be multiplied by sex and
              functions of age and year, i.e.

              .. code-block:: python

                βyearagesex1 * f1(age) * g1(year) * sex + βyearagesex2 * f1(age) * g2(year) * sex +
                βyearagesex3 * f2(age) * g1(year) * sex + βyearagesex4 * f2(age) * g2(year) * sex +
                βyearagesex5 * f3(age) * g1(year) * sex + βyearagesex6 * f3(age) * g2(year) * sex +
                βyearagesex7 * f4(age) * g1(year) * sex + βyearagesex8 * f4(age) * g2(year) * sex +
                βyearagesex9 * f5(age) * g1(year) * sex + βyearagesex10 * f5(age) * g2(year) * sex

            * ``βfam_hist (list[float])``: An array of 2 parameters to be multiplied by functions of
              age. See ``log_OR_family_history``.
            * ``βabx_exp (list[float])``: An array of 3 parameters to be multiplied by functions of
                age and antibiotic exposure. See ``log_OR_abx_exposure``.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = [
            "β0", "βsex", "βage", "βyear", "βsexage", "βsexyear", "βyearage",
            "βsexyearage", "βfam_hist", "βabx_exp"
        ]
        for key in KEYS:
            if key not in parameters.keys():
                raise ValueError(f"Missing key {key} in parameters.")
        self._parameters = parameters

    def load_occurrence_correction_table(self) -> DataFrameGroupBy:
        grouped_df = super().load_occurrence_correction_table(occurrence_type="prev")
        return grouped_df

    def crude_occurrence(self, sex: bool, age: int, year: int) -> float:
        poly_year = self.poly_year_calculator(year)
        poly_age = self.poly_age_calculator(age)
        poly_yearage = np.outer(poly_year, poly_age).flatten()
        return np.exp(
            self.parameters["β0"] +
            self.parameters["βsex"] * sex +
            np.dot(self.parameters["βyear"], poly_year) +
            np.dot(self.parameters["βage"], poly_age) +
            np.dot(self.parameters["βsexyear"], poly_year) * sex +
            np.dot(self.parameters["βsexage"], poly_age) * sex +
            np.dot(self.parameters["βyearage"], poly_yearage) +
            np.dot(self.parameters["βsexyearage"], poly_yearage) * sex
        )


def compute_asthma_age(
    agent: Agent,
    incidence: Incidence,
    prevalence: Prevalence,
    current_age: int,
    max_asthma_age: int = 110
) -> int:
    """Compute the age at which the person (agent) is first diagnosed with asthma.

    Args:
        agent: A person in the model.
        incidence: Asthma incidence.
        prevalence: Asthma prevalence.
        current_age: The current age of the agent.
    """
    # obtain the previous incidence
    min_year = incidence.min_year
    max_year = incidence.max_year

    if current_age == 3:
        return 3
    else:
        find_asthma_age = True
        asthma_age = 3
        year = min(max(agent.year - current_age + asthma_age, min_year), max_year)
        while find_asthma_age and asthma_age < max_asthma_age:
            has_asthma = agent_has_asthma(
                agent=agent,
                occurrence_type="inc",
                incidence=incidence,
                prevalence=prevalence,
                age=asthma_age,
                year=year
            )
            if has_asthma:
                return asthma_age
            asthma_age += 1
            asthma_age = min(asthma_age, incidence.max_age)
            year += 1
            year = min(year, max_year)
        return asthma_age


def agent_has_asthma(
    agent: Agent,
    occurrence_type: str,
    prevalence: Prevalence,
    incidence: Incidence | None = None,
    age: int | None = None,
    year: int | None = None,
) -> bool:
    """Determine whether the agent obtains a new asthma diagnosis based on age and sex.

    Args:
        agent: A person in the model.
        incidence: Asthma incidence.
        prevalence: Asthma prevalence.
        age: The age of the agent.
        year: The calendar year.
    """
    if occurrence_type == "inc" and incidence is None:
        raise ValueError("Incidence must be provided for incidence calculations.")

    if age is None:
        if occurrence_type == "inc":
            age = min(agent.age, incidence.max_age)
        else:
            age = min(agent.age - 1, prevalence.max_age)
    if year is None:
        if occurrence_type == "inc":
            year = agent.year
        else:
            year = agent.year - 1

    if age < 3:
        has_asthma = False
    elif age == 3:
        has_asthma = bool(np.random.binomial(1, prevalence.equation(
            sex=int(agent.sex), age=age, year=year, has_family_history=agent.has_family_history,
            dose=agent.num_antibiotic_use
        ))) # type: ignore
    elif age > 3 and occurrence_type == "inc":
        has_asthma = bool(
            np.random.binomial(
                n=1,
                p=incidence.equation(
                    int(agent.sex), age, year, agent.has_family_history, agent.num_antibiotic_use
                )
            ) # type: ignore
        ) 
    elif age > 3 and occurrence_type == "prev":
        has_asthma = bool(np.random.binomial(1, prevalence.equation(
            int(agent.sex), age, year, agent.has_family_history, agent.num_antibiotic_use
        ))) # type: ignore
    return has_asthma