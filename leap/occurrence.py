from __future__ import annotations
import abc
import copy
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from scipy.special import logit, expit
from leap.utils import get_data_path, poly, get_time_interval_tag
from leap.logger import get_logger
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.agent import Agent
    from leap.utils import Sex

logger = get_logger(__name__)

MIN_ASTHMA_AGE = 3
MAX_ABX_AGE = 7


class Occurrence:
    """A class containing information about asthma occurrence (incidence or prevalence)."""

    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        poly_parameters: Dict[str, list[float]] | None = None,
        max_age: int = 110,
        correction_table: DataFrameGroupBy | None = None,
        time_interval: dt.timedelta | relativedelta = relativedelta(years=1)
    ):
        if config is not None:
            self.parameters = config["parameters"]
            self.max_age = config["max_age"]
            self.poly_parameters = config["poly_parameters"]
        elif parameters is not None and poly_parameters is not None:
            self.parameters = parameters
            self.max_age = max_age
            self.poly_parameters = poly_parameters
        else:
            raise ValueError(
                "Either config dict or parameters and poly_parameters must be provided."
            )

        if correction_table is None:
            self.correction_table = self.load_occurrence_correction_table(
                time_interval=time_interval
            )
        else:
            self.correction_table = correction_table

        timepoints = np.unique([key[0] for key in self.correction_table.groups.keys()])
        self.min_timepoint = min(timepoints) + time_interval
        self.max_timepoint = max(timepoints)

    @property
    def parameters(self) -> dict:
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        self._parameters = copy.deepcopy(parameters)

    @property
    def correction_table(self) -> DataFrameGroupBy:
        """A dataframe grouped by timepoint, age, and sex.
        
        Each dataframe contains the following columns:

        * ``timepoint (datetime)``: the timepoint, e.g. ``2024-01-01``.
        * ``sex (str)``: ``F`` = female, ``M`` = male.
        * ``age (int)``: integer age.
        * ``correction (float)``: The correction term for the occurrence equation.

        See ``asthma_occurrence_correction.csv``.
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
    def min_timepoint(self) -> dt.datetime:
        return self._min_timepoint
    
    @min_timepoint.setter
    def min_timepoint(self, min_timepoint: dt.datetime):
        self._min_timepoint = min_timepoint

    @property
    def max_timepoint(self) -> dt.datetime:
        return self._max_timepoint
    
    @max_timepoint.setter
    def max_timepoint(self, max_timepoint: dt.datetime):
        self._max_timepoint = max_timepoint

    def load_occurrence_correction_table(
        self, occurrence_type: str, time_interval: dt.timedelta | relativedelta
    ) -> DataFrameGroupBy:
        """Load the asthma incidence correction table.

        Returns:
            Dataframe grouped by timepoint, age, and sex.
            Each dataframe contains the following columns:

            * ``timepoint (datetime)``: the timepoint, e.g. ``2024-01-01``.
            * ``sex (str)``: ``"F"`` = female, ``"M"`` = male.
            * ``age (int)``: integer age.
            * ``correction (float)``: The correction term for the occurrence equation.

        """
        time_interval_tag = get_time_interval_tag(time_interval)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_interval_tag}/asthma_occurrence_correction.csv"),
            parse_dates=["timepoint"]
        )
        df = df[df["type"] == occurrence_type]
        df.drop(columns=["type"], inplace=True)
        grouped_df = df.groupby(["timepoint", "sex", "age"])
        return grouped_df

    def equation(
        self, sex: Sex, age: int, timepoint: dt.datetime, has_family_history: bool, dose: int
    ) -> float:
        r"""Compute the asthma incidence / prevalence for a given risk factor combination.

        .. math::

            \zeta_{\lambda} = \sigma(\beta_{\eta} + \log(\omega_{\lambda}) - \alpha)

        where:

        * :math:`\beta_{\eta} = \sigma^{-1}(\eta(t))` is determined by the output of Model 1
        * :math:`\omega_{\lambda}` is the odds ratio for asthma incidence / prevalence based on
          family history and antibiotic doses
        * :math:`\alpha` is the incidence / prevalence correction term

        Args:
            sex: The sex of the agent.
            age: The age of the agent.
            timepoint: The calendar year.
            has_family_history: ``True`` if one or more parents of the agent has asthma,
                otherwise ``False``.
            dose: The number of courses of antibiotics taken during the first year of life.
        """
        correction_timepoint = min(timepoint, self.max_timepoint)
        timepoint = min(timepoint, self.max_timepoint)

        # Calculate asthma incidence / prevalence based on Model 1
        β_eta = self.crude_occurrence(sex, age, timepoint)

        # Calculate the odds ratio for asthma incidence / prevalence based on family history
        odds_ratio_fhx = self.calculate_odds_ratio_fam_history(age, has_family_history)

        # Calculate the odds ratio for asthma incidence / prevalence based on antibiotic doses
        odds_ratio_abx = self.calculate_odds_ratio_abx(age, dose)

        # Get the incidence or prevalence correction term
        α = self.correction_table.get_group(
            (correction_timepoint, str(sex), min(age, 63))
        )["correction"].values[0]

        p = expit(
            logit(β_eta) + np.log(odds_ratio_fhx) + np.log(odds_ratio_abx) + α
        )
        return p

    @abc.abstractmethod
    def crude_occurrence(self, sex: Sex, age: int, timepoint: dt.datetime) -> float:
        return

    def calculate_odds_ratio_fam_history(self, age: int, fam_hist: int) -> float:
        r"""Calculate the odds ratio for asthma prevalence based on family history.

        .. math::

            \log(\omega(f_{\lambda})) = 
                \beta_{\text{fhx_0}} \cdot f + 
                \beta_{\text{fhx_age}} \cdot (\text{min}(a, 5) - 3) \cdot f

        where:

        * :math:`\beta_{\text{fhx_xxx}}` is a constant coefficient
        * :math:`a` is the age
        * :math:`f` is the family history of asthma; 1 = at least one parent has asthma,
          0 = neither parent has asthma

        Args:
            age: The age of the individual in years.
            fam_hist: The family history of asthma, an integer in ``[0, 1]``, where 1 indicates
                at least one parent has asthma.

        Returns:
            The odds ratio for asthma prevalence based on family history and age.
        """
        β_fam_hist = self.parameters["β_fam_hist"]

        if fam_hist == 0:
            return 1.0

        return np.exp(
            β_fam_hist["β_fhx_0"] +
            β_fam_hist["β_fhx_age"] * (min(5, age) - MIN_ASTHMA_AGE)
        )

    def calculate_odds_ratio_abx(self, age: int, dose: int) -> float:
        r"""Calculate the odds ratio for asthma prevalence based on antibiotic exposure.

        .. math::

            \log(\omega(d_{\lambda})) =
                \begin{cases}
                \beta_{\text{abx_0}} + 
                \beta_{\text{abx_age}} \cdot \text{min}(a, 7) +
                \beta_{\text{abx_dose}} \cdot \text{min}(d, 3)
                && d > 0 \text{ and } a \leq 7 \\ \\
                0 && \text{otherwise}
                \end{cases}

        where:

        * :math:`\beta_{\text{abx_xxx}}` is a constant coefficient
        * :math:`a` is the age
        * :math:`d` is the number of courses of antibiotics taken during the first year of life

        Args:
            age: The age of the individual in years.
            dose: The number of antibiotic courses taken in the first year of life,
                an integer in ``[0, 5]``, where 5 indicates 5 or more courses.

        Returns:
            The odds ratio for asthma prevalence based on antibiotic exposure and age.
        """
        β_abx = self.parameters["β_abx"]

        if age > MAX_ABX_AGE or dose == 0:
            return 1.0
        else:
            return np.exp(
                β_abx["β_abx_0"] +
                β_abx["β_abx_age"] * min(age, MAX_ABX_AGE) +
                β_abx["β_abx_dose"] * min(dose, MIN_ASTHMA_AGE)
            )


class Incidence(Occurrence):
    """A class containing information about asthma incidence."""

    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        poly_parameters: Dict[str, list[float]] | None = None,
        max_age: int = 110,
        correction_table: DataFrameGroupBy | None = None,
        time_interval: dt.timedelta | relativedelta = relativedelta(years=1)
    ):
        super().__init__(config, parameters, poly_parameters, max_age, correction_table, time_interval)
        self.parameters["βage"] = np.array(self.parameters["βage"])
        self.parameters["βsexage"] = np.array(self.parameters["βsexage"])

    @property
    def parameters(self) -> dict:
        r"""A dictionary containing the following keys:
        
            * ``β0 (float)``: A constant parameter.
            * ``βsex (float)``: The parameter for the sex term, i.e.
              :math:`\beta_{\text{sex}} * \text{sex}`.
            * ``βtime (float)``: The parameter for the time term, i.e.
              :math:`\beta_{\text{time}} * \text{time}`.
            * ``βsextime (float)``: The parameter to be multiplied by sex and time, i.e.
              :math:`\beta_{\text{sextime}} * \text{time} * \text{sex}`.
            * ``βage (list[float])``: An array of 5 parameters to be multiplied by powers of age,
              i.e.

              .. math::

                \beta_{\text{age}1} * \text{age} + 
                \beta_{\text{age}2} * \text{age}^2 + 
                \beta_{\text{age}3} * \text{age}^3 +
                \beta_{\text{age}4} * \text{age}^4 +
                \beta_{\text{age}5} * \text{age}^5
            
            * ``βsexage (list[float])``: An array of 5 parameters to be multiplied by the sex and
              powers of age, i.e.

              .. math::

                \beta_{\text{sexage}1} * \text{sex} * \text{age} + 
                \beta_{\text{sexage}2} * \text{sex} * \text{age}^2 + 
                \beta_{\text{sexage}3} * \text{sex} * \text{age}^3 + \\
                \beta_{\text{sexage}4} * \text{sex} * \text{age}^4 +
                \beta_{\text{sexage}5} * \text{sex} * \text{age}^5

            * ``β_fam_hist (dict)``: A dictionary of 2 parameters to be multiplied by functions of
              age. See ``calculate_odds_ratio_fam_history``.
            * ``β_abx (dict)``: A dictionary of 3 parameters to be multiplied by functions of
              age and antibiotic dose. See ``calculate_odds_ratio_abx``.

        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["β0", "βsex", "βage", "βtime", "βsexage", "βsextime", "β_fam_hist", "β_abx"]
        for key in KEYS:
            if key not in parameters.keys():
                raise ValueError(f"Missing key {key} in parameters.")
        self._parameters = copy.deepcopy(parameters)

    @property
    def poly_parameters(self) -> Dict[str, list[float]]:
        r"""A dictionary containing the following keys:
        
            * ``alpha_age (list[float])``: The alpha vector from the normalization of the training
              data in the ``poly`` function. Length = degree of age polynomial = 5.
            * ``norm2_age (list[float])``: The :math:`\text{norm}^2` vector from the normalization
              of the training data in the ``poly`` function.
              Length = degree of age polynomial + 1 = 6.
        """
        return self._poly_parameters
    
    @poly_parameters.setter
    def poly_parameters(self, poly_parameters: Dict[str, list[float]]):
        KEYS = ["alpha_age", "norm2_age"]
        for key in KEYS:
            if key not in poly_parameters.keys():
                raise ValueError(f"Missing key {key} in poly_parameters.")
        self._poly_parameters = copy.deepcopy(poly_parameters)

    def load_occurrence_correction_table(self, time_interval: dt.timedelta | relativedelta) -> DataFrameGroupBy:
        """Load the asthma incidence correction table.

        Returns:
            A dataframe grouped by timepoint, age, and sex.
            Each dataframe contains the following columns:

            * ``timepoint (datetime)``: timepoint.
            * ``sex (str)``: ``"F"`` = female, ``"M"`` = male.
            * ``age (int)``: integer age.
            * ``correction (float)``: The correction term for the asthma incidence / prevalence
              equation.

        """
        grouped_df = super().load_occurrence_correction_table(
            occurrence_type="incidence", time_interval=time_interval
        )
        return grouped_df

    def crude_occurrence(
        self,
        sex: Sex,
        age: int,
        timepoint: dt.datetime
    ) -> float:
        r"""Calculate the crude asthma incidence.

        .. math::

            \eta^{(i)} = 
                \sum_{m=0}^1 \beta_{01m} t^{(i)} \cdot (s^{(i)})^m +
                \sum_{k=0}^{5} \sum_{m=0}^{1} \beta_{k0m} \cdot (a^{(i)})^k \cdot (s^{(i)})^m


        where:

        * :math:`\eta^{(i)}` is the crude asthma incidence
        * :math:`\beta_{k\ell m}` is the coefficient for the feature
          :math:`(a^{(i)})^k \cdot (t^{(i)})^{\ell} \cdot (s^{(i)})^m`
        * :math:`a^{(i)}` is the age
        * :math:`t^{(i)}` is the year
        * :math:`s^{(i)}` is the sex

        Args:
            sex: The sex of the agent.
            age: The age of the agent.
            timepoint: The calendar year.

        Returns:
            A float representing the crude asthma incidence for the given timepoint, age, and sex.
        """

        poly_age = poly(
            age,
            degree=5,
            alpha=self.poly_parameters["alpha_age"],
            norm2=self.poly_parameters["norm2_age"]
        ).flatten()
        return np.exp(
            self.parameters["β0"] +
            self.parameters["βsex"] * int(sex) +
            self.parameters["βtime"] * timepoint.year +
            self.parameters["βsextime"] * int(sex) * timepoint.year +
            np.dot(self.parameters["βage"], poly_age) +
            np.dot(self.parameters["βsexage"], poly_age) * int(sex)
        )


class Prevalence(Occurrence):
    """A class containing information about asthma prevalence."""

    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        poly_parameters: Dict[str, list[float]] | None = None,
        max_age: int = 110,
        correction_table: DataFrameGroupBy | None = None,
        time_interval: dt.timedelta | relativedelta = relativedelta(years=1)
    ):
        super().__init__(
            config, parameters, poly_parameters, max_age, correction_table, time_interval
        )
        self.parameters["βage"] = np.array(self.parameters["βage"])
        self.parameters["βsexage"] = np.array(self.parameters["βsexage"])
        self.parameters["βsextime"] = np.array(self.parameters["βsextime"])
        self.parameters["βtimeage"] = np.array(self.parameters["βtimeage"])
        self.parameters["βsextimeage"] = np.array(self.parameters["βsextimeage"])

    @property
    def parameters(self) -> dict:
        r"""A dictionary containing the following keys:

            * ``β0 (float)``: A constant parameter, determined by Model 1.
            * ``βsex (float)``: The parameter for the sex term, i.e.
              :math:`\beta_{\text{sex}} * \text{sex}`.
            * ``βtime (list[float])``: An array of 2 parameters to be multiplied by powers of year,
              i.e.
              
              .. math::
              
                \beta_{\text{time}1} * \text{time} + \beta_{\text{time}2} * \text{time}^2

            * ``βsextime (list[float])``: An array of 2 parameters to be multiplied by sex and
              powers of time, i.e.

              .. math::
              
                \beta_{\text{sextime}1} * \text{sex} * \text{time} + 
                \beta_{\text{sextime}2} * \text{sex} * \text{time}^2

            * ``βage (list[float])``: An array of 5 parameters to be multiplied by powers of age,
              i.e.

              .. math::

                \beta_{\text{age}1} * \text{age} + 
                \beta_{\text{age}2} * \text{age}^2 + 
                \beta_{\text{age}3} * \text{age}^3 +
                \beta_{\text{age}4} * \text{age}^4 +
                \beta_{\text{age}5} * \text{age}^5

            * ``βsexage (list[float])``: An array of 5 parameters to be multiplied by the sex and
              powers of age, i.e.

              .. math::

                \beta_{\text{sexage}1} * \text{sex} * \text{age} + 
                \beta_{\text{sexage}2} * \text{sex} * \text{age}^2 + 
                \beta_{\text{sexage}3} * \text{sex} * \text{age}^3 + \\
                \beta_{\text{sexage}4} * \text{sex} * \text{age}^4 +
                \beta_{\text{sexage}5} * \text{sex} * \text{age}^5

            * ``βtimeage (list[float])``: An array of 10 parameters to be multiplied by powers of
              age and time, i.e.

              .. math::

                \beta_{\text{timeage}1} * \text{time} * \text{age} +
                \beta_{\text{timeage}2} * \text{time}^2 * \text{age} + \\
                \beta_{\text{timeage}3} * \text{time} * \text{age}^2 +
                \beta_{\text{timeage}4} * \text{time}^2 * \text{age}^2 + \\
                \beta_{\text{timeage}5} * \text{time} * \text{age}^3 +
                \beta_{\text{timeage}6} * \text{time}^2 * \text{age}^3 + \\
                \beta_{\text{timeage}7} * \text{time} * \text{age}^4 +
                \beta_{\text{timeage}8} * \text{time}^2 * \text{age}^4 + \\
                \beta_{\text{timeage}9} * \text{time} * \text{age}^5 +
                \beta_{\text{timeage}10} * \text{time}^2 * \text{age}^5

            * ``βsextimeage (list[float])``: An array of 10 parameters to be multiplied by sex and
              powers of age and time, i.e.

              .. math::

                \beta_{\text{sextimeage}1} * \text{sex} * \text{time} * \text{age} +
                \beta_{\text{sextimeage}2} * \text{sex} * \text{time}^2 * \text{age} + \\
                \beta_{\text{sextimeage}3} * \text{sex} * \text{time} * \text{age}^2 +
                \beta_{\text{sextimeage}4} * \text{sex} * \text{time}^2 * \text{age}^2 + \\
                \beta_{\text{sextimeage}5} * \text{sex} * \text{time} * \text{age}^3 +
                \beta_{\text{sextimeage}6} * \text{sex} * \text{time}^2 * \text{age}^3 + \\
                \beta_{\text{sextimeage}7} * \text{sex} * \text{time} * \text{age}^4 +
                \beta_{\text{sextimeage}8} * \text{sex} * \text{time}^2 * \text{age}^4 + \\
                \beta_{\text{sextimeage}9} * \text{sex} * \text{time} * \text{age}^5 +
                \beta_{\text{sextimeage}10} * \text{sex} * \text{time}^2 * \text{age}^5

            * ``β_fam_hist (dict)``: A dictionary of 2 parameters to be multiplied by functions of
              age. See ``calculate_odds_ratio_fam_history``.
            * ``β_abx (dict)``: A dictionary of 3 parameters to be multiplied by functions of
              age and antibiotic dose. See ``calculate_odds_ratio_abx``.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = [
            "β0", "βsex", "βage", "βtime", "βsexage", "βsextime", "βtimeage",
            "βsextimeage", "β_fam_hist", "β_abx"
        ]
        for key in KEYS:
            if key not in parameters.keys():
                raise ValueError(f"Missing key {key} in parameters.")
        self._parameters = copy.deepcopy(parameters)

    @property
    def poly_parameters(self) -> Dict[str, list[float]]:
        r"""A dictionary containing the following keys:
        
            * ``alpha_age (list[float])``: The alpha vector from the normalization of the training
              data in the ``poly`` function. Length = degree of age polynomial = 5.
            * ``norm2_age (list[float])``: The :math:`\text{norm}^2` vector from the normalization
              of the training data in the ``poly`` function.
              Length = degree of age polynomial + 1 = 6.
            * ``alpha_time (list[float])``: The alpha vector from the normalization of the training
              data in the ``poly`` function. Length = degree of time polynomial = 2.
            * ``norm2_time (list[float])``: The :math:`\text{norm}^2` vector from the normalization
              of the training data in the ``poly`` function.
              Length = degree of time polynomial + 1 = 3.
        """
        return self._poly_parameters
    
    @poly_parameters.setter
    def poly_parameters(self, poly_parameters: Dict[str, list[float]]):
        KEYS = ["alpha_age", "norm2_age", "alpha_time", "norm2_time"]
        for key in KEYS:
            if key not in poly_parameters.keys():
                raise ValueError(f"Missing key {key} in poly_parameters.")
        self._poly_parameters = copy.deepcopy(poly_parameters)

    def load_occurrence_correction_table(
        self, time_interval: dt.timedelta | relativedelta
    ) -> DataFrameGroupBy:
        grouped_df = super().load_occurrence_correction_table(
            occurrence_type="prevalence", time_interval=time_interval
        )
        return grouped_df

    def crude_occurrence(
        self,
        sex: Sex,
        age: int,
        timepoint: dt.datetime
    ) -> float:
        r"""Calculate the crude asthma prevalence.

        .. math::

            \eta^{(i)} = \sum_{k=0}^{5} \sum_{\ell=0}^2 \sum_{m=0}^1 \beta_{k \ell m} 
                \cdot (a^{(i)})^k \cdot (t^{(i)})^{\ell} \cdot (s^{(i)})^m

        where:

        * :math:`\beta_{k\ell m}` is the coefficient for the feature
          :math:`(a^{(i)})^k \cdot (t^{(i)})^{\ell} \cdot (s^{(i)})^m`
        * :math:`a^{(i)}` is the age
        * :math:`t^{(i)}` is the year
        * :math:`s^{(i)}` is the sex

        There are :math:`6 * 3 * 2 = 36` coefficients in the prevalence model.
        
        Args:
            sex: The sex of the agent.
            age: The age of the agent.
            timepoint: The timepoint.

        Returns:
            A float representing the crude asthma prevalence for the given timepoint, age, and sex.
        """

        poly_year = poly(
            timepoint.year,
            degree=2,
            alpha=self.poly_parameters["alpha_time"],
            norm2=self.poly_parameters["norm2_time"]
        ).flatten()
        poly_age = poly(
            age,
            degree=5,
            alpha=self.poly_parameters["alpha_age"],
            norm2=self.poly_parameters["norm2_age"]
        ).flatten()
        poly_yearage = np.outer(poly_year, poly_age).flatten()
        return np.exp(
            self.parameters["β0"] +
            self.parameters["βsex"] * int(sex) +
            np.dot(self.parameters["βtime"], poly_year) +
            np.dot(self.parameters["βage"], poly_age) +
            np.dot(self.parameters["βsextime"], poly_year) * int(sex) +
            np.dot(self.parameters["βsexage"], poly_age) * int(sex) +
            np.dot(self.parameters["βtimeage"], poly_yearage) +
            np.dot(self.parameters["βsextimeage"], poly_yearage) * int(sex)
        )


def compute_asthma_age(
    agent: Agent,
    incidence: Incidence,
    prevalence: Prevalence,
    current_age: int,
    max_asthma_age: int = 110,
    time_interval: dt.timedelta | relativedelta = relativedelta(years=1)
) -> int:
    """Compute the age at which the person (agent) is first diagnosed with asthma.

    Args:
        agent: A person in the model.
        incidence: Asthma incidence.
        prevalence: Asthma prevalence.
        current_age: The current age of the agent.
    """
    # obtain the previous incidence
    min_timepoint = incidence.min_timepoint
    max_timepoint = incidence.max_timepoint

    if current_age == MIN_ASTHMA_AGE:
        return MIN_ASTHMA_AGE
    else:
        find_asthma_age = True
        asthma_age = dt.timedelta(days=365 * MIN_ASTHMA_AGE)
        timepoint = min(
            max(
                agent.timepoint - relativedelta(years=current_age) + asthma_age,
                min_timepoint
            ),
            max_timepoint
        )
        while find_asthma_age and asthma_age < dt.timedelta(days=365 * max_asthma_age):
            has_asthma = agent_has_asthma(
                agent=agent,
                occurrence_type="incidence",
                incidence=incidence,
                prevalence=prevalence,
                age=asthma_age.days // 365,
                timepoint=timepoint
            )
            if has_asthma:
                return asthma_age.days // 365
            asthma_age += time_interval
            asthma_age = min(asthma_age, dt.timedelta(days=365 * incidence.max_age))
            year += 1
            year = min(year, max_timepoint)
        return asthma_age.days // 365


def agent_has_asthma(
    agent: Agent,
    occurrence_type: str,
    prevalence: Prevalence,
    incidence: Incidence | None = None,
    age: int | None = None,
    timepoint: dt.datetime | None = None,
    time_interval: dt.timedelta | relativedelta = relativedelta(years=1)
) -> bool:
    """Determine whether the agent obtains a new asthma diagnosis based on age and sex.

    Args:
        agent: A person in the model.
        occurrence_type: One of ``"incidence"`` or ``"prevalence"``.
        prevalence: Asthma prevalence object.
        incidence: Asthma incidence object.
        age: The age of the agent.
        timepoint: The timepoint.
    """
    if occurrence_type == "incidence" and incidence is None:
        raise ValueError("Incidence must be provided for incidence calculations.")

    if age is None:
        if occurrence_type == "incidence":
            age = min(agent.age, incidence.max_age)
        else:
            age = min(agent.age - 1, prevalence.max_age)
    if timepoint is None:
        if occurrence_type == "incidence":
            timepoint = agent.timepoint
        else:
            timepoint = agent.timepoint - time_interval

    if age < MIN_ASTHMA_AGE:
        has_asthma = False
    elif age == MIN_ASTHMA_AGE:
        has_asthma = bool(np.random.binomial(1, prevalence.equation(
            sex=agent.sex,
            age=age,
            timepoint=timepoint,
            has_family_history=agent.has_family_history,
            dose=agent.num_antibiotic_use
        ))) # type: ignore
    elif age > MIN_ASTHMA_AGE and occurrence_type == "incidence":
        has_asthma = bool(
            np.random.binomial(
                n=1,
                p=incidence.equation(
                    agent.sex,
                    age,
                    timepoint,
                    agent.has_family_history,
                    agent.num_antibiotic_use
                )
            ) # type: ignore
        ) 
    elif age > MIN_ASTHMA_AGE and occurrence_type == "prevalence":
        has_asthma = bool(np.random.binomial(1, prevalence.equation(
            agent.sex, age, timepoint, agent.has_family_history, agent.num_antibiotic_use
        ))) # type: ignore
    return has_asthma