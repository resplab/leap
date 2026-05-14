from __future__ import annotations
import copy
import pandas as pd
import numpy as np
import datetime as dt
from leap.utils import get_data_path, check_province, get_time_delta_tag, TimeDelta
from leap.control import ControlLevels
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from leap.agent import Agent
    from leap.utils import Sex
    from dateutil.relativedelta import relativedelta


class ExacerbationHistory:
    """A class containing information about the history of asthma exacerbations.

    Attributes:
        num_current_timepoint: the number of exacerbations at the current timepoint.
        num_prev_timepoint: the number of exacerbations at the previous timepoint.
    """
    def __init__(self, num_current_timepoint: int, num_prev_timepoint: int):
        self.num_current_timepoint = num_current_timepoint
        self.num_prev_timepoint = num_prev_timepoint


class Exacerbation:
    """A class containing information about asthma exacerbations."""

    def __init__(
        self,
        config: dict | None = None,
        province: str = "CA",
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        calibration_table: DataFrameGroupBy | None = None,
        initial_rate: float | None = None,
        time_delta: dt.timedelta | relativedelta | TimeDelta = TimeDelta(years=1)
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
            self.calibration_table = self.load_exacerbation_calibration(province, time_delta)
        else:
            self.calibration_table = calibration_table

        self.time_delta = time_delta
        self.assign_random_β0()
        self.parameters["min_timepoint"] = min(
            [key[0] for key in self.calibration_table.groups.keys()]
        ) + time_delta

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
        self._hyperparameters = copy.deepcopy(hyperparameters)

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:

        * ``β0``: float, a constant parameter, randomly selected from a normal distribution
          with mean ``β0_μ`` and standard deviation ``β0_σ``. See ``hyperparameters``.
        * ``βcontrol_C``: float, the parameter for the controlled asthma term.
        * ``βcontrol_PC``: float, the parameter for the partially-controlled asthma term.
        * ``βcontrol_UC``: float, the parameter for the uncontrolled asthma term.
        * ``min_timepoint``: int, the minimum timepoint for which exacerbation data exists + 1.
          Currently 2001.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = [
            "βcontrol_C", "βcontrol_PC", "βcontrol_UC"
        ]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"Missing key {key} in parameters.")
        self._parameters = copy.deepcopy(parameters)

    @property
    def calibration_table(self) -> DataFrameGroupBy:
        r"""A dataframe grouped by timepoint and sex, with the following columns:

        * ``timepoint (int)``: a datetime timepoint.
        * ``sex (str)``: ``F`` = female, ``M`` = male.
        * ``age (int)``: integer age.
        * ``calibrator_multiplier (float)``: A multiplier used to calibrate the exacerbation rate;
          used to compute the :math:`\lambda` parameter for the Poisson distribution:
        
        .. math::

            \lambda = \alpha \cdot e^{\beta_0} \prod_{i=1}^3 e^{\beta_i c_i} 

        See ``exacerbation_calibration.csv``.
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

    def load_exacerbation_calibration(
        self, province: str, time_delta: dt.timedelta | relativedelta | TimeDelta
    ) -> DataFrameGroupBy:
        r"""Load the exacerbation calibration table.

        Args:
            province: A string indicating the province abbreviation, e.g. ``"BC"``.
                For all of Canada, set province to ``"CA"``.
            time_delta: A time interval for the calibration data, e.g. ``relativedelta(years=1)``.

        Returns:
            A dataframe grouped by timepoint and sex.

            Each entry is a dataframe with the following columns:

            * ``timepoint (int)``: a datetime timepoint.
            * ``sex (str)``: ``F`` = female, ``M`` = male.
            * ``age (int)``: integer age.
            * ``calibrator_multiplier (float)``: A multiplier used to calibrate the exacerbation rate;
              used to compute the :math:`\lambda` parameter for the Poisson distribution:
        
            .. math::

                \lambda = \alpha \cdot e^{\beta_0} \prod_{i=1}^3 e^{\beta_i c_i} 
        """
        time_delta_tag = get_time_delta_tag(time_delta)
        df = pd.read_csv(
            get_data_path(f"processed_data/{time_delta_tag}/exacerbation_calibration.csv"),
            parse_dates=["timepoint"]
        )
        check_province(province)

        df = df[df["province"] == province]
        df.drop(["province"], axis=1, inplace=True)
        grouped_df = df.groupby(["timepoint", "sex"])
        return grouped_df

    def compute_num_exacerbations(
        self,
        agent: Agent | None = None,
        age: int | None = None,
        sex: Sex | int | None = None,
        timepoint: dt.datetime | None = None,
        control_levels: ControlLevels | None = None,
        initial: bool = False
    ) -> int:
        """Compute the number of asthma exacerbations in a given time interval.

        Args:
            agent: A person in the model.
            age: The age of the person in years.
            sex: The sex of the agent (person), 0 = female, 1 = male.
            timepoint: The starting date / time of the time interval to compute the number of
                exacerbations in, e.g. ``2024-01-01``.
            control_levels: The asthma control levels.
            initial: If this is the initial computation.

        Returns:
            The number of asthma exacerbations in the given time interval.
        """

        if agent is not None:
            age = agent.age
            sex = int(agent.sex)
            timepoint = agent.timepoint
            control_levels = agent.control_levels
        elif age is None or sex is None or timepoint is None or control_levels is None:
            raise ValueError("Either agent or age, sex, timepoint, and control_levels must be provided.")

        if initial:
            timepoint = max(self.parameters["min_timepoint"], timepoint - self.time_delta)
            age = min(age - 1, 90)
            if age < 3:
                return 0
        else:
            timepoint = max(self.parameters["min_timepoint"], timepoint)
            age = min(age, 90)

        df = self.calibration_table.get_group((timepoint, int(sex)))
        calibrator_multiplier = df[df["age"] == age]["calibrator_multiplier"]

        μ = (
            self.parameters["β0"] +
            control_levels.uncontrolled * self.parameters["βcontrol_UC"] +
            control_levels.partially_controlled * self.parameters["βcontrol_PC"] +
            control_levels.fully_controlled * self.parameters["βcontrol_C"] +
            np.log(calibrator_multiplier)
        )
        λ = np.exp(μ)
        return np.random.poisson(λ)[0]
