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

    This refers to how well the condition is managed.
    There are three levels of asthma control:
        fully-controlled = 1
        partially-controlled = 2
        uncontrolled = 3

    Attributes:
        hyperparameters (dict): A dictionary containing the hyperparameters used
            to compute `β0` from a normal distribution:
            * ``β0_μ``: float, the mean of the normal distribution.
            * ``β0_σ``: float, the standard deviation of the normal distribution.
        parameters (dict): A dictionary containing the following keys:
            * ``β0``: float, a constant parameter. See `hyperparameters`.
            * ``βage``: float, the parameter for the age term.
            * ``βsex``: float, the parameter for the sex term.
            * ``βsexage``: float, the parameter for the sex * age term.
            * ``βsexage2``: float, the parameter for the sex * age^2 term.
            * ``βage2``: float, the parameter for the age^2 term.
            * ``βDx2``: float, unused?
            * ``βDx3``: float, unused?
            * ``θ``: list of two numbers, which are used as the thresholds to compute the ordinal
              regression.
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
        elif parameters is not None and hyperparameters is not None:
            self.hyperparameters = hyperparameters
            self.parameters = parameters
            self.assign_random_β0()
        else:
            raise ValueError(
                "Either config dict or parameters and hyperparameters must be provided."
            )

    def assign_random_β0(self):
        """Assign the parameter β0 a random value from a normal distribution."""
        self.parameters["β0"] = np.random.normal(
            self.hyperparameters["β0_μ"],
            self.hyperparameters["β0_σ"]
        )

    def compute_control_levels(self, sex: Sex, age: int, initial: bool = False) -> ControlLevels:
        """Compute the probability that the control level = k for each value of k.

        The probability is given by ordinal regression, where y = control level:

        .. math::

            P(y <= k) = σ(θ_k - η)
            P(y == k) = P(y <= k) - P(y < k + 1)
                      = σ(θ_k - η) - σ(θ_(k+1) - η)

        Args:
            sex: Sex of person, 1 = male, 0 = female.
            age: The age of the person (agent) in years.
            initial: If this is the initial computation.

        Returns:
            ControlLevels: A ``ControlLevel`` object with the probability of each control level.
                For example:
                {
                    "fully_controlled": 0.2,
                    "partially_controlled": 0.75,
                    "uncontrolled": 0.05
                }

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
