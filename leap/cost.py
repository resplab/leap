import numpy as np
from leap.logger import get_logger

logger = get_logger(__name__)


class AsthmaCost:
    """
    A class containing information about the cost of asthma.

    Attributes:
        parameters (dict): A dictionary containing the following keys:
            * ``control``: A vector of numbers.
            * ``exacerbation``: A vector of numbers.

    """
    def __init__(self, config: dict | None = None, parameters: dict | None = None):
        if config is None and parameters is None:
            raise ValueError("Either config dict or parameters must be provided.")
        elif config is not None:
            self.parameters = config["parameters"]
            exchange_rate_usd_cad = config["exchange_rate_usd_cad"]
            self.parameters["exac"] = np.array(self.parameters["exac"]) * exchange_rate_usd_cad
            self.parameters["control"] = np.array(
                self.parameters["control"]
            ) * exchange_rate_usd_cad

    def compute_cost(self, agent) -> float:
        """Compute the cost in dollars for the current year due to asthma exacerbations and control.

        If ``control_levels`` are not present, will default to equal probability of
        fully controlled, partially controlled, and uncontrolled asthma.

        Args:
            agent (Agent): A person in the model.
        """

        if not agent.has_asthma:
            return 0.0
        else:
            control_levels = agent.control_levels

            return (
                np.dot(
                    agent.exacerbation_severity_history.current_year,
                    self.parameters["exac"]
                ) +
                np.dot(control_levels.as_array(), self.parameters["control"])
            )
