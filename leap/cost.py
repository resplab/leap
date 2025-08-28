from __future__ import annotations
import numpy as np
from leap.logger import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.agent import Agent

logger = get_logger(__name__)


class AsthmaCost:
    """A class containing information about the dollar cost of asthma."""

    def __init__(
        self,
        config: dict | None = None,
        exac: np.ndarray | list[float] | None = None,
        control_probs: np.ndarray | list[float] | None = None,
        exchange_rate_usd_cad: float | None = None
    ):
        if config is not None:
            parameters = config["parameters"].copy()
            self.exchange_rate_usd_cad = config["exchange_rate_usd_cad"]
            self.exac = np.array(parameters["exac"]) * self.exchange_rate_usd_cad
            self.control_probs = np.array(
                parameters["control"]
            ) * self.exchange_rate_usd_cad
        elif exac is not None and exchange_rate_usd_cad is not None and control_probs is not None:
            self.exac = np.array(exac) * exchange_rate_usd_cad
            self.control_probs = np.array(control_probs) * exchange_rate_usd_cad
        else:
            raise ValueError(
                "Either config dict or control_probs, exac, and exchange rate must be provided."
            )
    @property
    def exchange_rate_usd_cad(self) -> float:
        """The exchange rate from USD to CAD."""
        return self._exchange_rate_usd_cad

    @exchange_rate_usd_cad.setter
    def exchange_rate_usd_cad(self, exchange_rate_usd_cad: float):
        self._exchange_rate_usd_cad = exchange_rate_usd_cad

    @property
    def exac(self) -> np.ndarray:
        """A vector of 4 factors to multiply by the 4 exacerbation severity levels."""
        return self._exac

    @exac.setter
    def exac(self, exac: np.ndarray):
        self._exac = exac

    @property
    def control_probs(self) -> np.ndarray:
        """A vector of 3 factors to multiply by the 3 control level probabilities."""
        return self._control_probs

    @control_probs.setter
    def control_probs(self, control_probs: np.ndarray):
        self._control_probs = control_probs

    def compute_cost(self, agent: Agent) -> float:
        """Compute the cost in dollars for the current year due to asthma exacerbations and control.

        Args:
            agent: A person in the model.

        Returns:
            The cost in Canadian dollars for the current year due to asthma exacerbations and
            control.

        Examples:

            >>> from leap.agent import Agent
            >>> from leap.control import ControlLevels
            >>> from leap.severity import ExacerbationSeverityHistory
            >>> from leap.cost import AsthmaCost
            >>> agent = Agent(
            ...     sex="F",
            ...     age=30,
            ...     year=2027,
            ...     year_index=0,
            ...     control_levels=ControlLevels(0.1, 0.2, 0.7),
            ...     exacerbation_severity_history=ExacerbationSeverityHistory(
            ...         current_year=[0, 1, 5, 6], prev_year=[0, 0, 0, 0]
            ...     ),
            ...     has_family_history=True,
            ...     has_asthma=True,
            ...     num_antibiotic_use=0
            ... )
            >>> asthma_cost = AsthmaCost(
            ...     control_probs=[2372, 2965, 3127],
            ...     exac=[130, 594, 2425, 9900],
            ...     exchange_rate_usd_cad=1.25
            ... )
            >>> cost = asthma_cost.compute_cost(agent)
            >>> print(f"Total cost in 2027 for female aged 30: ${cost:.2f} CAD")
            Total cost in 2027 for female aged 30: $93922.62 CAD

        """

        if not agent.has_asthma:
            return 0.0
        else:
            control_levels = agent.control_levels

            return (
                np.dot(
                    agent.exacerbation_severity_history.current_year,
                    self.exac
                ) +
                np.dot(control_levels.as_array(), self.control_probs)
            )
