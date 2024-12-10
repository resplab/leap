from __future__ import annotations
import pandas as pd
import numpy as np
import pathlib
from leap.utils import PROCESSED_DATA_PATH
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from leap.utils import Sex


class AntibioticExposure:
    """A class containing information about antibiotic use.

    Attributes:
        parameters: A dictionary containing the following keys:
            * ``β0``: float, the constant parameter when computing μ.
            * ``βyear``: float, the parameter to be multiplied by the agent's birth year for
              computing μ.
            * ``β2005``: float, an added constant parameter if the agent's birth year > 2005 for
              computing μ.
            * ``βsex``: float, the parameter to be multiplied by the agent's sex when computing μ.
            * ``θ``: int, the number of successes (the r parameter) in the negative binomial
              distribution.
            * ``β2005_year``: float, if the agent's birth year is > 2005, ``β2005_year`` will be
              multiplied by the birth year when computing μ.
            * ``fixyear``: integer or nothing. If present, replaces the ``year`` parameter when
              computing the probability for the negative binomial distribution.
            * ``βfloor``: float, the minimum value of μ.

        mid_trends: A set of data frames grouped by year and sex.
            Each entry is a DataFrame with a single row with the following columns:

            * ``year``: integer
            * ``sex``: 0 = female, 1 = male
            * ``rate``: float, TODO.

    """
    def __init__(
        self,
        config: dict | None = None,
        parameters: dict | None = None,
        mid_trends: pd.api.typing.DataFrameGroupBy | None = None
    ):
        if config is not None:
            self.parameters = config["parameters"]
        elif parameters is not None:
            self.parameters = parameters
        else:
            raise ValueError("Either config dict or parameters must be provided.")

        if mid_trends is None:
            self.mid_trends = self.load_abx_mid_trends()
        else:
            self.mid_trends = mid_trends

    @property
    def parameters(self) -> dict:
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: dict):
        KEYS = ["β0", "βyear", "β2005", "βsex", "θ", "β2005_year", "fixyear", "βfloor"]
        for key in KEYS:
            if key not in parameters:
                raise ValueError(f"Key {key} not found in parameters.")
        self._parameters = parameters

    def load_abx_mid_trends(self):
        """Load the antibiotic mid trends table.

        Returns:
            pd.api.typing.DataFrameGroupBy: a set of data frames grouped by year and sex.
            Each entry is a DataFrame with a single row with the following columns:

                * ``year``: integer
                * ``sex``: 0 = female, 1 = male
                * ``rate``: float, TODO.
        """
        df = pd.read_csv(pathlib.Path(PROCESSED_DATA_PATH, "midtrends.csv"))
        grouped_df = df.groupby(["year", "sex"])
        return grouped_df

    def compute_num_antibiotic_use(self, sex: Sex | int, birth_year: int) -> int:
        """Compute the number of antibiotics used during the first year of life.

        Args:
            sex: Sex of agent, 1 = male, 0 = female.
            birth_year: The year the agent (person) was born.
        """
        if birth_year < 2001:
            p = self.compute_probability(sex=sex, year=2000)
        elif self.parameters["fixyear"] is not None:
            if isinstance(self.parameters["fixyear"], (int, float)):
                p = self.compute_probability(
                    sex=sex,
                    year=int(self.parameters["fixyear"])
                )
            else:
                μ = max(
                    self.mid_trends.get_group((birth_year, int(sex)))["rate"].iloc[0],
                    self.parameters["βfloor"]
                )
                p = self.parameters["θ"] / (self.parameters["θ"] + μ)
        else:
            p = self.compute_probability(
                sex=sex,
                year=birth_year
            )
        r = self.parameters["θ"]
        return np.random.negative_binomial(r, p)

    def compute_probability(self, sex: Sex | int, year: int) -> float:
        """Compute the probability of antibiotic exposure for a given year and sex.

        Args:
            sex: Sex of agent, 1 = male, 0 = female.
            year: The calendar year.
        """
        μ = np.exp(
            self.parameters["β0"] +
            self.parameters["βsex"] * int(sex) +
            self.parameters["βyear"] * year +
            self.parameters["β2005"] * (year > 2005) +
            self.parameters["β2005_year"] * (year > 2005) * year
        )
        μ = max(μ, self.parameters["βfloor"] / 1000)
        return self.parameters["θ"] / (self.parameters["θ"] + μ)
