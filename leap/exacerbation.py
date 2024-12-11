from __future__ import annotations
import pandas as pd
import numpy as np
from leap.utils import get_data_path
from leap.control import ControlLevels
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.agent import Agent
    from leap.utils import Sex


class ExacerbationHistory:
    """A class containing information about the history of asthma exacerbations.

    Attributes:
        num_current_year: the number of exacerbations in the current year.
        num_prev_year: the number of exacerbations in the previous year.
    """
    def __init__(self, num_current_year: int, num_prev_year: int):
        self.num_current_year = num_current_year
        self.num_prev_year = num_prev_year


class Exacerbation:
    """A class containing information about asthma exacerbations."""

    def __init__(
        self,
        config: dict | None = None,
        province: str = "CA",
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        calibration_table: DataFrameGroupBy | None = None,
        initial_rate: float | None = None
    ):
        if config is not None:
            self.hyperparameters = config["hyperparameters"]
            self.parameters = config["parameters"]
            self.initial_rate = config["initial_rate"]
        elif parameters is not None and hyperparameters is not None:
            self.hyperparameters = hyperparameters
            self.parameters = parameters
            self.initial_rate = initial_rate
        else:
            raise ValueError(
                "Either config dict or parameters and hyperparameters must be provided."
            )
        
        if calibration_table is None:
            self.calibration_table = self.load_exacerbation_calibration(province)
        else:
            self.calibration_table = calibration_table

        self.assign_random_β0()
        self.parameters["min_year"] = min(
            [key[0] for key in self.calibration_table.groups.keys()]
        ) + 1

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
            if key not in hyperparameters:
                raise ValueError(f"Missing key {key} in hyperparameters.")
        self._hyperparameters = hyperparameters

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:
            * ``β0``: float, a constant parameter, randomly selected from a normal distribution
              with mean ``β0_μ`` and standard deviation ``β0_σ``. See ``hyperparameters``.
            * ``β0_calibration``: float, the parameter for the calibration term.
            * ``βage``: float, the parameter for the age term.
            * ``βsex``: float, the parameter for the sex term.
            * ``βasthmaDx``: float, TODO.
            * ``βprev_exac1``: float, TODO.
            * ``βprev_exac2``: float, TODO.
            * ``βcontrol``: float, the parameter for the asthma control term.
            * ``βcontrol_C``: float, the parameter for the controlled asthma term.
            * ``βcontrol_PC``: float, the parameter for the partially-controlled asthma term.
            * ``βcontrol_UC``: float, the parameter for the uncontrolled asthma term.
            * ``min_year``: int, the minimum year for which exacerbation data exists + 1.
              Currently 2001.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = [
            "β0_calibration", "βage", "βsex", "βasthmaDx", "βprev_exac1", "βprev_exac2",
            "βcontrol", "βcontrol_C", "βcontrol_PC", "βcontrol_UC"
        ]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"Missing key {key} in parameters.")
        self._parameters = parameters

    @property
    def calibration_table(self) -> DataFrameGroupBy:
        """A dataframe grouped by year and sex, with the following columns:

            * ``year``: integer year.
            * ``sex``: 1 = male, 0 = female.
            * ``age``: integer age.
            * ``calibrator_multiplier``: float, TODO.

            See ``master_calibrated_exac.csv``.
        """
        return self._calibration_table
    
    @calibration_table.setter
    def calibration_table(self, calibration_table: DataFrameGroupBy):
        self._calibration_table = calibration_table

    def assign_random_β0(self):
        """Assign the parameter β0 a random value from a normal distribution."""
        self.parameters["β0"] = np.random.normal(
            self.hyperparameters["β0_μ"],
            self.hyperparameters["β0_σ"]
        )

    def load_exacerbation_calibration(self, province: str) -> DataFrameGroupBy:
        """Load the exacerbation calibration table.

        Args:
            province: A string indicating the province abbreviation, e.g. "BC".
                For all of Canada, set province to "CA".

        Returns:
            A dataframe grouped by year and sex, with the following columns:

                * ``year``: integer year.
                * ``sex``: 1 = male, 0 = female.
                * ``age``: integer age.
                * ``calibrator_multiplier``: float, TODO.
        """

        df = pd.read_csv(
            get_data_path("processed_data", "master_calibrated_exac.csv")
        )
        df = df[df["province"] == province]
        df.drop(["province"], axis=1, inplace=True)
        grouped_df = df.groupby(["year", "sex"])
        return grouped_df

    def compute_num_exacerbations(
        self,
        agent: Agent | None = None,
        age: int | None = None,
        sex: Sex | int | None = None,
        year: int | None = None,
        control_levels: ControlLevels | None = None,
        initial: bool = False
    ) -> int:
        """Compute the number of asthma exacerbations in a given year.

        Args:
            agent: A person in the model.
            age: The age of the person in years.
            sex: The sex of the agent (person), 0 = female, 1 = male.
            year: The calendar year, e.g. 2024.
            control_levels: The asthma control levels.
            initial: If this is the initial computation.

        Returns:
            The number of asthma exacerbations.
        """

        if agent is not None:
            age = agent.age
            sex = int(agent.sex)
            year = agent.year
            control_levels = agent.control_levels
        elif age is None or sex is None or year is None or control_levels is None:
            raise ValueError("Either agent or age, sex, year, and control_levels must be provided.")

        if initial:
            year = max(self.parameters["min_year"], year - 1)
            age = min(age - 1, 90)
            if age < 3:
                return 0
        else:
            year = max(self.parameters["min_year"], year)
            age = min(age, 90)

        df = self.calibration_table.get_group((year, int(sex)))
        calibrator_multiplier = df[df["age"] == age]["calibrator_multiplier"]

        μ = (
            self.parameters["β0"] +
            int(not initial) * self.parameters["β0_calibration"] +
            age * self.parameters["βage"] +
            int(sex) * self.parameters["βsex"] +
            control_levels.uncontrolled * self.parameters["βcontrol_UC"] +
            control_levels.partially_controlled * self.parameters["βcontrol_PC"] +
            control_levels.fully_controlled * self.parameters["βcontrol_C"] +
            np.log(calibrator_multiplier)
        )
        λ = np.exp(μ)
        return np.random.poisson(λ)[0]
