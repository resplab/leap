import pandas as pd
import numpy as np
from leap.logger import get_logger
from typing import Tuple

logger = get_logger(__name__)


PROVINCE_MAP = {
    "Canada": "CA",
    "British Columbia": "BC",
    "Alberta": "AB",
    "Saskatchewan": "SK",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
    "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Prince Edward Island": "PE",
    "Yukon": "YT",
    "Northwest Territories": "NT",
    "Nunavut": "NU"
}


def get_province_id(province: str) -> str:
    """Convert full length province name to abbreviation.
    
    Args:
        province: The full length province name, e.g. ``British Columbia``.
        
    Returns:
        The abbreviation for the province, e.g. ``BC``.
    """
    return PROVINCE_MAP[province]


def get_sex_id(sex: str) -> str:
    """Convert full length sex to single character.
    
    Args:
        sex: The full length string, either ``Female`` or ``Male``
    
    Returns:
        The single character string, either ``F`` or ``M``.
    """
    return sex[0:1]

def parse_age_group(x: str, max_age: int) -> Tuple[int, int]:
    """Parse an age group string into a tuple of integers.
    
    Args:
        x: The age group string. Must be in the format "X-Y", "X+", "X-Y years", "<1 year".

    Returns:
        A tuple of integers representing the lower and upper age of the age group.

    Examples:
    
        >>> parse_age_group("0-4", max_age=65)
        (0, 4)
        >>> parse_age_group("5-9 years", max_age=65)
        (5, 9)
        >>> parse_age_group("10+", max_age=65)
        (10, 65)
        >>> parse_age_group("<1 year", max_age=65)
        (0, 1)
    """
    if x == "<1 year":
        return 0, 1
    elif "-" in x:
        return int(x.split(" ")[0].split("-")[0]), int(x.split(" ")[0].split("-")[1])
    elif "+" in x:
        return int(x.split("+")[0]), max_age
    else:
        raise ValueError(f"Invalid age group: {x}")
    

def format_age_group(age_group: str, upper_age_group: str = "100 years and over") -> int:
    """Convert age group to integer.
    
    Args:
        age_group: The age group string, e.g. ``5 to 9 years``.
        upper_age_group: The upper age group string, e.g. ``100 years and over``.
        
    Returns:
        The integer age.

    Examples:

    >>> format_age_group("110 years and over", "110 years and over")
    110
    >>> format_age_group("Under 1 year", "100 years and over")
    0
    >>> format_age_group("9 years")
    9
    """
    if age_group == upper_age_group:
        age = age_group.replace(" years and over", "")
        age = int(age)
    elif age_group == "Under 1 year":
        age = 0
    else:
        age = age_group.replace(" years", "")
        age = age.replace(" year", "")
        age = int(age)
    return age


def heaviside(x: float | list[float] | np.ndarray | pd.Series, threshold: float) -> int | list[int]:
    """Heaviside step function.
    
    Args:
        x: The input value or array of values.
        threshold: The threshold value.
        
    Returns:
        1 if ``x >= threshold``, else 0. If ``x`` is a vector, this is computed for each entry.
    """

    if isinstance(x, float) or isinstance(x, (int, np.integer)):
        return 1 if x >= threshold else 0
    else:
        return [1 if i >= threshold else 0 for i in x]


class ContingencyTable:
    """A class representing a contingency table."""
    def __init__(self, a: float, b: float, c: float, d: float):
        """Initialize the contingency table with proportions.
        Args:
            a: Proportion of the population with variable 1 + and variable 2 +.
            b: Proportion of the population with variable 1 + and variable 2 -.
            c: Proportion of the population with variable 1 - and variable 2 +.
            d: Proportion of the population with variable 1 - and variable 2 -.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self):
        return f"ContingencyTable(values={self.a}, {self.b}, {self.c}, {self.d})"
    
    def to_list(self) -> list[float]:
        """Convert the contingency table to a list of values."""
        return [self.a, self.b, self.c, self.d]
    
    def apply(self, func) -> "ContingencyTable":
        """Apply a function to each value in the contingency table."""
        return ContingencyTable(
            a=func(self.a),
            b=func(self.b),
            c=func(self.c),
            d=func(self.d)
        )
    

def conv_2x2(
    ori: float,
    ni: float,
    n1i: float,
    n2i: float,
    var_names: list = ["ai", "bi", "ci", "di"],
) -> ContingencyTable:
    r"""Create a 2x2 contigency table.
    
    This function is based off the ``R`` function ``metafor::conv.2x2``.

    We want to determine the contingency table:

    .. raw:: html

        <table class="table">
            <thead>
            <tr>
                <th></th>
                <th>variable 2, outcome +</th>
                <th>variable 2, outcome -</th>
                <th></th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td>variable 1, outcome +</td>
                <td><code class="notranslate">ai</code></td>
                <td><code class="notranslate">bi</code></td>
                <td><code class="notranslate">n1i</code></td>
            </tr>
            <tr>
                <td>variable 1, outcome -</td>
                <td><code class="notranslate">ci</code></td>
                <td><code class="notranslate">di</code></td>
                <td></td>
            </tr>
            <tr>
                <td></td>
                <td><code class="notranslate">n2i</code></td>
                <td></td>
                <td><code class="notranslate">ni</code></td>
            </tr>
            </tbody>
        </table>

    Given the odds ratio :math:`or_{i}`, the marginal counts :math:`n_{1i}` and :math:`n_{2i}`,
    and the total sample size :math:`n_{i}`, we want to compute the probabilities
    :math:`a_{i}`, :math:`b_{i}`, :math:`c_{i}`, and :math:`d_{i}`.

    .. math::

        n_{i} &= a_{i} + b_{i} + c_{i} + d_{i} \\
        n_{1i} &= a_{i} + b_{i} \\
        n_{2i} &= a_{i} + c_{i} \\
        or_{i} &= \dfrac{a_{i} d_{i}}{b_{i} c_{i}}
    
    Args:
        ori: The odds ratio.
        ni: The total sample size.
        n1i: The marginal count for the first variable.
        n2i: The marginal count for the second variable.
        var_names: The names of the variables. Must be of length 4.
        
    Returns:
        A pandas DataFrame with the cell frequencies for the 2x2 table.

    Examples:

        Let's suppose we know that the probability of antibiotic use in infancy is 0.52,
        and the probability of having an asthma diagnosis is 0.87, and suppose we have `100`
        people. We also know that the odds ratio, i.e. the odds of getting asthma given
        antibiotic exposure, is ``ori=0.4343``. Then the contingency table would be:

        .. raw:: html

            <table class="table">
                <thead>
                <tr>
                    <th></th>
                    <th>asthma</th>
                    <th>no asthma</th>
                    <th></th>
                </tr>
                </thead>
                <tbody>
                <tr>
                    <td>antibiotics</td>
                    <td><code class="notranslate">ai</code></td>
                    <td><code class="notranslate">bi</code></td>
                    <td><code class="notranslate">n1i = 52</code></td>
                </tr>
                <tr>
                    <td>no antibiotics</td>
                    <td><code class="notranslate">ci</code></td>
                    <td><code class="notranslate">di</code></td>
                    <td></td>
                </tr>
                <tr>
                    <td></td>
                    <td><code class="notranslate">n2i = 87</code></td>
                    <td></td>
                    <td><code class="notranslate">ni = 100</code></td>
                </tr>
                </tbody>
            </table>


        We want to compute ``ai``, ``bi``, ``ci``, and ``di``. We can do this using the
        ``conv_2x2`` function:

        >>> from leap.data_generation.utils import conv_2x2
        >>> conv_2x2(ori=0.4343, ni=100, n1i=52, n2i=87)
        ContingencyTable(values=43, 9, 44, 4)

        Here we have:

        * ``ai`` = 43, the number of people who have asthma and were exposed to antibiotics.
        * ``bi`` = 9, the number of people who have asthma and were not exposed to antibiotics.
        * ``ci`` = 44, the number of people who do not have asthma and were exposed to antibiotics.
        * ``di`` = 4, the number of people who do not have asthma and were not exposed to antibiotics.

        We can divide them by ``ni`` to get the proportions:

        * ``ai`` = 0.43, the probability of having asthma given antibiotic exposure.
        * ``bi`` = 0.09, the probability of having asthma given no antibiotic exposure.
        * ``ci`` = 0.44, the probability of not having asthma given antibiotic exposure.
        * ``di`` = 0.04, the probability of not having asthma given no antibiotic exposure.
    """

    if len(var_names) != 4:
        raise ValueError("Argument 'var.names' must be of length 4.")

    ni = int(np.round(ni, 0))
    n1i = int(np.round(n1i, 0))
    n2i = int(np.round(n2i, 0))

    if ni < 0 or n1i < 0 or n2i < 0:
        raise ValueError("One or more sample sizes or marginal counts are negative.")

    if n1i > ni or n2i > ni:
        raise ValueError("One or more marginal counts are larger than the sample sizes.")

   # compute marginal proportions for the two variables

    p1i = n1i / ni
    p2i = n2i / ni

    x = ori * (p1i + p2i) + (1 - p1i) - p2i
    y = np.sqrt(x**2 - 4 * p1i * p2i * ori * (ori - 1))

    p11i = (x - y) / (2 * (ori - 1))

    ai = int(np.round(ni * p11i, 0))
    bi = n1i - ai
    ci = n2i - ai
    di = ni - ai - bi - ci

    df = pd.DataFrame({
        "values": [ai, bi, ci, di],
    }, index=var_names)

    # check for negative cell frequencies
    df["has_neg"] = df["values"].apply(lambda x: x < 0)
    has_neg = df["has_neg"].any()
    if has_neg:
        logger.warning("There are negative cell frequencies in the table.")
        df["values"] = df["values"].apply(lambda x: np.nan if x < 0 else x)

 
    df.drop(columns=["has_neg"], inplace=True)

    return ContingencyTable(
        a=df.loc[var_names[0]].values[0],
        b=df.loc[var_names[1]].values[0],
        c=df.loc[var_names[2]].values[0],
        d=df.loc[var_names[3]].values[0]
    )