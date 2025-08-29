import copy
import numpy as np
from leap.logger import get_logger

logger = get_logger(__name__)


class FamilyHistory:
    """A class containing information about family history of asthma."""

    def __init__(self, config: dict | None = None, probability: float | None = None):
        if config is not None:
            self.probability = config["parameters"]["p"]
        elif probability is not None:
            self.probability = probability
        else:
            raise ValueError("Either config dict or probability must be provided.")

    @property
    def probability(self) -> float:
        """The probability that an agent has a family history of asthma."""
        return self._probability

    @probability.setter
    def probability(self, probability: float):
        if probability > 1 or probability < 0:
            raise ValueError(
                f"p must be a probability between 0 and 1, received {probability}."
            )
        self._probability = probability

    def has_family_history_of_asthma(self) -> bool:
        """Use Bernoulli distribution to determine whether an agent has a family history of asthma.

        Returns:
            Whether or not an agent has a family history of asthma.

        Examples:

            >>> from leap.family_history import FamilyHistory
            >>> family_history = FamilyHistory(probability=1.0)
            >>> family_history.has_family_history_of_asthma()
            True

        """
        return bool(np.random.binomial(1, self.probability))

    def __copy__(self):
        return FamilyHistory(probability=self.probability)
    
    def __deepcopy__(self):
        return FamilyHistory(probability=copy.deepcopy(self.probability))

    def copy(self, deep: bool = True):
        if deep:
            return self.__deepcopy__()
        else:
            return self.__copy__()
