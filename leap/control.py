from __future__ import annotations
import numpy as np
from leap.utils import compute_ordinal_regression
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.utils import Sex


class ControlLevels:
    """A class containing the probability of each control level.

    Attributes:
        fully_controlled: The probability of being fully controlled.
        partially_controlled: The probability of being partially controlled.
        uncontrolled: The probability of being uncontrolled.
    """
    def __init__(
        self, fully_controlled: float, partially_controlled: float, uncontrolled: float
    ):
        self.fully_controlled = fully_controlled
        self.partially_controlled = partially_controlled
        self.uncontrolled = uncontrolled

    def as_array(self) -> np.ndarray:
        """Return the control levels as an array."""
        return np.array([self.fully_controlled, self.partially_controlled, self.uncontrolled])


class Control:
    """A class containing information about asthma control.

    This refers to how well the condition is managed. There are three levels of asthma control:

        * 1 = fully-controlled
        * 2 = partially-controlled
        * 3 = uncontrolled

    """
    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        hyperparameters: dict | None = None
    ):
        if config is not None:
            self.hyperparameters = config["hyperparameters"]
            self.parameters = config["parameters"]
            self.parameters["θ"] = list(self.parameters["θ"])
            self.assign_random_β0()
        elif parameters is not None:
            self.hyperparameters = hyperparameters
            self.parameters = parameters
            if "β0" not in self.parameters:
                self.assign_random_β0()
        else:
            raise ValueError(
                "Either config dict or parameters and hyperparameters must be provided."
            )
        
    @property
    def hyperparameters(self) -> dict | None:
        """Hyperparameters used to compute ``β0`` from a normal distribution:
            * ``β0_μ``: float, the mean of the normal distribution.
            * ``β0_σ``: float, the standard deviation of the normal distribution.
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters: dict | None):
        if hyperparameters is not None:
            KEYS = ["β0_μ", "β0_σ"]
            for key in KEYS:
                if key not in hyperparameters:
                    raise ValueError(f"Missing key {key} in hyperparameters.")
        self._hyperparameters = hyperparameters

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:
            * ``β0``: float, a constant parameter. See ``hyperparameters``.
            * ``βage``: float, the parameter for the ``age`` term.
            * ``βsex``: float, the parameter for the ``sex`` term.
            * ``βsexage``: float, the parameter for the ``sex * age`` term.
            * ``βsexage2``: float, the parameter for the ``sex * age^2`` term.
            * ``βage2``: float, the parameter for the ``age^2`` term.
            * ``βDx2``: float, unused?
            * ``βDx3``: float, unused?
            * ``θ``: list of two numbers, which are used as the thresholds to compute the ordinal
              regression.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["βage", "βsex", "βsexage", "βsexage2", "βage2", "βDx2", "βDx3", "θ"]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"Missing key {key} in parameters.")
        self._parameters = parameters

    def assign_random_β0(self):
        """Assign the parameter β0 a random value from a normal distribution."""
        if self.hyperparameters is None:
            raise ValueError("Hyperparameters must be provided.")

        self.parameters["β0"] = np.random.normal(
            self.hyperparameters["β0_μ"],
            self.hyperparameters["β0_σ"]
        )

    def compute_control_levels(
        self, sex: Sex | int, age: int, initial: bool = False
    ) -> ControlLevels:
        """Compute the probability that the control level = ``k`` for each value of ``k``.

        The probability is given by ordinal regression, where y = control level:

        .. math::

            P(y <= k) = σ(θ_k - η)
            P(y == k) = P(y <= k) - P(y < k + 1)
                      = σ(θ_k - η) - σ(θ_(k+1) - η)

        Args:
            sex: Sex of person, 1 = male, 0 = female.
            age: The age of the person (agent) in years.
            initial: Whether or not this is the initial computation.

        Returns:
            ControlLevels: A ``ControlLevel`` object with the probability of each control level.

        Examples:

            >>> from leap.control import Control
            >>> from leap.utils import get_data_path
            >>> import json
            >>> with open(get_data_path("processed_data/config.json")) as f:
            ...     config = json.load(f)["control"]
            >>> config["parameters"]["θ"] = [-10**5, -10**5] 
            >>> control = Control(config=config)
            >>> control_levels = control.compute_control_levels(sex=0, age=45)
            >>> control_levels.as_array()
            array([0., 0., 1.])
            >>> control_levels.fully_controlled
            np.float64(0.0)
            >>> control_levels.partially_controlled
            np.float64(0.0)
            >>> control_levels.uncontrolled
            np.float64(1.0)

        """
        if initial:
            age_scaled = (age - 1) / 100
        else:
            age_scaled = age / 100

        η = (
            self.parameters["β0"] +
            age_scaled * self.parameters["βage"] +
            int(sex) * self.parameters["βsex"] +
            age_scaled * int(sex) * self.parameters["βsexage"] +
            age_scaled ** 2 * int(sex) * self.parameters["βsexage2"] +
            age_scaled ** 2 * self.parameters["βage2"]
        )

        control_levels_prob = compute_ordinal_regression(η, self.parameters["θ"])
        control_levels = ControlLevels(
            fully_controlled=control_levels_prob[0],
            partially_controlled=control_levels_prob[1],
            uncontrolled=control_levels_prob[2]
        )
        return control_levels
