from __future__ import annotations
import numpy as np
from leap.antibiotic_exposure import AntibioticExposure
from leap.census_division import CensusDivision
from leap.control import ControlLevels
from leap.exacerbation import ExacerbationHistory
from leap.pollution import Pollution
from leap.severity import ExacerbationSeverityHistory
from leap.utils import UUID4, Sex
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.family_history import FamilyHistory


class Agent:
    """A person in the model.

    Attributes:
        uuid:
            A unique identifier for the agent.
        sex:
            Sex of the agent, one of "M", "F".
        age:
            Age of the person in years.
        year:
            The calendar year of the current iteration, e.g. 2027.
        year_index:
            An integer representing the year of the simulation. For example, if
            the simulation starts in 2023, then the `year_index` for 2023 is 1, for 2024 is 2, etc.
        alive:
            Whether the person is alive, ``True`` = alive.
        num_antibiotic_use:
            TODO.
        has_asthma:
            Whether the person has asthma, ``True`` = has asthma.
        asthma_age:
            Age at which the person was diagnosed with asthma.
        severity:
            Asthma severity level: 1 = mild, 2 = severe, 3 = very severe.
        control_levels:
            Asthma control level:
            1 = fully controlled, 2 = partially controlled, 3 = uncontrolled.
        exacerbation_history:
            Total number of exacerbations.
        exacerbation_severity_history:
            Number of exacerbations by severity.
        total_hosp:
            Total number of very severe asthma exacerbations leading to hospitalization.
        has_family_history:
            Is there a family history of asthma?
        asthma_status:
            TODO.
        census_division:
            The Canadian census division where the agent resides.
        pollution:
            The pollution data for the agent's census division.
    """
    def __init__(
        self,
        sex: str | int | bool | Sex,
        age: int,
        year: int,
        year_index: int,
        month: int = 1,
        province: str = "CA",
        uuid: UUID4 = UUID4(),
        alive: bool = True,
        num_antibiotic_use: int | None = None,
        has_asthma: bool = False,
        asthma_age: int | None = None,
        asthma_status: bool = False,
        control_levels: ControlLevels = ControlLevels(0.3333, 0.3333, 0.3333),
        exacerbation_history: ExacerbationHistory = ExacerbationHistory(0, 0),
        exacerbation_severity_history: ExacerbationSeverityHistory = ExacerbationSeverityHistory(np.zeros(4), np.zeros(4)),
        total_hosp: int = 0,
        has_family_history: bool | None = None,
        family_history: FamilyHistory | None = None,
        antibiotic_exposure: AntibioticExposure | None = None,
        census_division: CensusDivision | None = None,
        pollution: Pollution | None = None,
        ssp: str = "SSP1_2.6"
    ):
        self.uuid = uuid
        self.sex = sex
        self.age = age
        self.year = year
        self.year_index = year_index
        self.month = month
        self.alive = alive
        self.has_asthma = has_asthma
        self.asthma_age = asthma_age
        self.asthma_status = asthma_status
        self.control_levels = control_levels
        self.exacerbation_history = exacerbation_history
        self.exacerbation_severity_history = exacerbation_severity_history
        self.total_hosp = total_hosp
        if num_antibiotic_use is None and antibiotic_exposure is not None:
            self.num_antibiotic_use = antibiotic_exposure.compute_num_antibiotic_use(
                sex=int(self.sex),
                birth_year=year - age
            )
        elif num_antibiotic_use is not None:
            self.num_antibiotic_use = num_antibiotic_use
        else:
            raise ValueError("Either num_antibiotic_use or antibiotic_exposure must be provided.")
        if has_family_history is None and family_history is not None:
            self.has_family_history = family_history.has_family_history_of_asthma()
        elif has_family_history is not None:
            self.has_family_history = has_family_history
        else:
            raise ValueError("Either has_family_history or family_history must be provided.")
        if census_division is None:
            self.census_division = CensusDivision(province=province, year=year)
        else:
            self.census_division = census_division
        if pollution is None:
            self.pollution = Pollution(self.census_division.cduid, year, month, ssp)
        else:
            self.pollution = pollution

    @property
    def census_division(self) -> CensusDivision:
        return self._census_division
    
    @census_division.setter
    def census_division(self, census_division: CensusDivision):
        self._census_division = census_division

    @property
    def exacerbation_history(self) -> ExacerbationHistory:
        return self._exacerbation_history
    
    @exacerbation_history.setter
    def exacerbation_history(self, exacerbation_history: ExacerbationHistory):
        self._exacerbation_history = exacerbation_history

    @property
    def has_family_history(self) -> bool:
        return self._has_family_history
    
    @has_family_history.setter
    def has_family_history(self, has_family_history: bool):
        self._has_family_history = has_family_history

    @property
    def num_antibiotic_use(self) -> int:
        return self._num_antibiotic_use
    
    @num_antibiotic_use.setter
    def num_antibiotic_use(self, num_antibiotic_use: int):
        self._num_antibiotic_use = num_antibiotic_use

    @property
    def sex(self) -> Sex:
        return self._sex
    
    @sex.setter
    def sex(self, value: Sex | str | int | bool):
        if isinstance(value, Sex):
            self._sex = value
        else:
            self._sex = Sex(value)
    