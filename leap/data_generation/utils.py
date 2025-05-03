import pandas as pd
import numpy as np
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

    if isinstance(x, float) or isinstance(x, int):
        return 1 if x >= threshold else 0
    else:
        return [1 if i >= threshold else 0 for i in x]
