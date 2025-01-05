import numpy as np
from leap.logger import get_logger

logger = get_logger(__name__)


class FamilyHistory:
    """A class containing information about family history of asthma."""

    def __init__(self, config: dict | None = None, parameters: dict | None = None):
        if config is not None:
            self.parameters = config["parameters"]
        elif parameters is not None:
            self.parameters = parameters
        else:
            raise ValueError("Either config dict or parameters must be provided.")

    @property
    def parameters(self) -> dict:
        """A dictionary containing the following keys:
            * ``p``: float, the probability that an agent has a family history of asthma.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict):
        if parameters["p"] > 1 or parameters["p"] < 0:
            raise ValueError(
                f"p must be a probability between 0 and 1, received {parameters['p']}."
            )
        self._parameters = parameters

    def has_family_history_of_asthma(self) -> bool:
        """Use Bernoulli distribution to determine whether an agent has a family history of asthma.

        Returns:
            Whether or not an agent has a family history of asthma.

        Examples:

            >>> from leap.family_history import FamilyHistory
            >>> family_history = FamilyHistory(parameters={"p": 1.0})
            >>> family_history.has_family_history_of_asthma()
            True

        """
        return bool(np.random.binomial(1, self.parameters["p"]))
