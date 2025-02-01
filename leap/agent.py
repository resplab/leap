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
        year:
            The calendar year of the current iteration, e.g. 2027.
        year_index:
            An integer representing the year of the simulation. For example, if
            the simulation starts in 2023, then the ``year_index`` for 2023 is 1, for 2024 is 2, etc.
    """
    def __init__(
        self,
        sex: str | int | bool | Sex,
        age: int,
        year: int,
        year_index: int,
        month: int = 1,
        province: str = "CA",
        uuid: UUID4 | None = None,
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
        ignore_pollution_flag: bool = False,
        pollution: Pollution | None = None,
        ssp: str = "SSP1_2.6"
    ):
        if uuid is None:
            uuid = UUID4()
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
        self.ignore_pollution_flag = ignore_pollution_flag
        if not self.ignore_pollution_flag:
            self.pollution = Pollution(self.census_division.cduid, year, month, ssp)
        else:
            self.pollution = pollution

    @property
    def age(self) -> int:
        """The age of the person in years."""
        return self._age

    @age.setter
    def age(self, age: int):
        self._age = age

    @property
    def alive(self) -> bool:
        """Whether or not the person is still alive."""
        return self._alive

    @alive.setter
    def alive(self, alive: bool):
        self._alive = alive

    @property
    def asthma_age(self) -> int | None:
        """The age at which the person was diagnosed with asthma."""
        return self._asthma_age
    
    @asthma_age.setter
    def asthma_age(self, asthma_age: int | None):
        self._asthma_age = asthma_age

    @property
    def asthma_status(self) -> bool:
        """TODO."""
        return self._asthma_status
    
    @asthma_status.setter
    def asthma_status(self, asthma_status: bool):
        self._asthma_status = asthma_status

    @property
    def census_division(self) -> CensusDivision:
        """The Canadian census division where the person resides."""
        return self._census_division
    
    @census_division.setter
    def census_division(self, census_division: CensusDivision):
        self._census_division = census_division

    @property
    def control_levels(self) -> ControlLevels:
        """The control levels for the person's asthma.
        
        This refers to how well the condition is managed. There are three levels of asthma control:

        * 1 = fully-controlled
        * 2 = partially-controlled
        * 3 = uncontrolled

        """
        return self._control_levels

    @control_levels.setter
    def control_levels(self, control_levels: ControlLevels):
        self._control_levels = control_levels

    @property
    def exacerbation_history(self) -> ExacerbationHistory:
        """The asthma exacerbation history of the person.
        
        The exacerbation history object contains the total number of asthma exacerbations for
        the current year (``num_current_year``) and the total number of asthma exacerbations
        for the previous year (``num_previous_year``).

        """
        return self._exacerbation_history
    
    @exacerbation_history.setter
    def exacerbation_history(self, exacerbation_history: ExacerbationHistory):
        self._exacerbation_history = exacerbation_history

    @property
    def exacerbation_severity_history(self) -> ExacerbationSeverityHistory:
        """The number of asthma exacerbations by severity.
        
        The exacerbation severity history object contains the number of asthma exacerbations
        by severity for the current year (``current_year``) and the number of asthma exacerbations
        by severity for the previous year (``previous_year``). There are 4 levels of severity:

        * 0 = mild
        * 1 = moderate
        * 2 = severe
        * 3 = very severe
        
        """
        return self._exacerbation_severity_history

    @exacerbation_severity_history.setter
    def exacerbation_severity_history(
        self, exacerbation_severity_history: ExacerbationSeverityHistory
    ):
        self._exacerbation_severity_history = exacerbation_severity_history

    @property
    def has_asthma(self) -> bool:
        """Whether or not the person has asthma."""
        return self._has_asthma

    @has_asthma.setter
    def has_asthma(self, has_asthma: bool):
        self._has_asthma = has_asthma

    @property
    def has_family_history(self) -> bool:
        """Whether or not the person has a family history of asthma."""
        return self._has_family_history
    
    @has_family_history.setter
    def has_family_history(self, has_family_history: bool):
        self._has_family_history = has_family_history

    @property
    def num_antibiotic_use(self) -> int:
        """The number of times the person has used a round of antibiotics."""
        return self._num_antibiotic_use
    
    @num_antibiotic_use.setter
    def num_antibiotic_use(self, num_antibiotic_use: int):
        self._num_antibiotic_use = num_antibiotic_use

    @property
    def pollution(self) -> Pollution:
        """The pollution data for the person's census division."""
        return self._pollution

    @pollution.setter
    def pollution(self, pollution: Pollution):
        self._pollution = pollution

    @property
    def province(self) -> str:
        """The province where the person resides.
        
        This is a two-letter abbreviation, e.g. ``BC`` for British Columbia:
        
        * ``CA``: All of Canada
        * ``AB``: Alberta
        * ``BC``: British Columbia
        * ``MB``: Manitoba
        * ``NB``: New Brunswick
        * ``NL``: Newfoundland and Labrador
        * ``NS``: Nova Scotia
        * ``NT``: Northwest Territories
        * ``NU``: Nunavut
        * ``ON``: Ontario
        * ``PE``: Prince Edward Island
        * ``QC``: Quebec
        * ``SK``: Saskatchewan
        * ``YT``: Yukon

        """
        return self._province
    
    @province.setter
    def province(self, province: str):
        self._province = province

    @property
    def sex(self) -> Sex:
        """The sex of the person."""
        return self._sex
    
    @sex.setter
    def sex(self, value: Sex | str | int | bool):
        if isinstance(value, Sex):
            self._sex = value
        else:
            self._sex = Sex(value)

    @property
    def ssp(self) -> str:
        """The shared socioeconomic pathway (SSP) scenario.
        
        Used for determining the pollution data in the agent's census division if the pollution
        data is not provided.
        """
        return self._ssp
    
    @ssp.setter
    def ssp(self, ssp: str):
        self._ssp = ssp

    @property
    def total_hosp(self) -> int:
        """The total number of very severe asthma exacerbations leading to hospitalization."""
        return self._total_hosp

    @total_hosp.setter
    def total_hosp(self, total_hosp: int):
        self._total_hosp = total_hosp

    @property
    def uuid(self) -> UUID4:
        """A unique identifier for the agent."""
        return self._uuid

    @uuid.setter
    def uuid(self, uuid: UUID4):
        self._uuid = uuid
    