import numpy as np
from leap.census_division import CensusDivision
from leap.exacerbation import ExacerbationHistory
from leap.pollution import Pollution
from leap.severity import ExacerbationSeverityHistory
from leap.utils import UUID4, Sex


class Agent:
    """A person in the model.

    Attributes:
        uuid (UUID4):
            A unique identifier for the agent.
        sex:
            Sex of the agent, one of "M", "F".
        age (int):
            Age of the person in years.
        year (int):
            The calendar year of the current iteration, e.g. 2027.
        year_index (int):
            An integer representing the year of the simulation. For example, if
            the simulation starts in 2023, then the `year_index` for 2023 is 1, for 2024 is 2, etc.
        alive (bool):
            Whether the person is alive, true = alive.
        num_antibiotic_use (int):
            TODO.
        has_asthma (bool):
            Whether the person has astham, True = has asthma.
        asthma_age (int):
            Age at which the person was diagnosed with asthma.
        severity (int):
            Asthma severity level: 1 = mild, 2 = severe, 3 = very severe.
        control_levels (ControlLevels):
            Asthma control level:
            1 = fully controlled, 2 = partially controlled, 3 = uncontrolled.
        exacerbation_history (ExacerbationHistory):
            Total number of exacerbations.
        exacerbation_severity_history (ExacerbationSeverityHistory):
            Number of exacerbations by severity.
        total_hosp (int):
            Total number of very severe asthma exacerbations leading to hospitalization.
        has_family_history (bool):
            Is there a family history of asthma?
        asthma_status (bool):
            TODO.
        census_division (CensusDivision):
            The Canadian census division where the agent resides.
        pollution (Pollution):
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
        asthma_age=None,
        asthma_status: bool = False,
        severity=None,
        control_levels=None,
        exacerbation_history=ExacerbationHistory(0, 0),
        exacerbation_severity_history=ExacerbationSeverityHistory(np.zeros(4), np.zeros(4)),
        total_hosp: int = 0,
        has_family_history: bool | None = None,
        family_history=None,
        antibiotic_exposure=None,
        census_division=None,
        pollution=None,
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
        self.severity = severity
        self.control_levels = control_levels
        self.exacerbation_history = exacerbation_history
        self.exacerbation_severity_history = exacerbation_severity_history
        self.total_hosp = total_hosp
        if num_antibiotic_use is None and antibiotic_exposure is not None:
            self.num_antibiotic_use = antibiotic_exposure.compute_num_antibiotic_use(
                sex=int(self.sex),
                birth_year=year - age
            )
        else:
            self.num_antibiotic_use = num_antibiotic_use
        if has_family_history is None and family_history is not None:
            self.has_family_history = family_history.has_family_history_of_asthma()
        else:
            self.has_family_history = has_family_history
        if census_division is None:
            self.census_division = CensusDivision(province=province, year=year)
        else:
            self.census_division = census_division
        if pollution is None:
            self.pollution = Pollution(self.census_division.cduid, year, month, ssp)
        else:
            self.pollution = pollution

    @property
    def sex(self) -> Sex:
        return self._sex
    
    @sex.setter
    def sex(self, value: Sex | str | int | bool):
        if isinstance(value, Sex):
            self._sex = value
        else:
            self._sex = Sex(value)
    