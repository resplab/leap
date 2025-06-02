import pandas as pd
import numpy as np
import itertools
from leap.utils import get_data_path
from leap.logger import get_logger
from leap.data_generation.occurrence_calibration_data import get_asthma_occurrence_prediction

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 1999
STABILIZATION_YEAR = 2025
MIN_ASTHMA_AGE = 3  # Minimum age for asthma diagnosis
MAX_ASTHMA_AGE = 62
MAX_AGE = 110

def get_asthma_df(
    starting_year: int = STARTING_YEAR,
    end_year: int = 2065,
    min_age: int = MIN_ASTHMA_AGE,
    max_age: int = MAX_AGE,
    max_asthma_age: int = MAX_ASTHMA_AGE,
    stabilization_year: int = STABILIZATION_YEAR
) -> pd.DataFrame:
    """Loads the asthma prevalence / incidence predictions from Model 1.

    Args:
        starting_year: The starting year for the dataframe.
        end_year: The ending year for the dataframe.
        min_age: The minimum age for asthma prediction.
        max_age: The maximum age for asthma prediction.
        max_asthma_age: The maximum age for for which the asthma prevalence / incidence
            model can accurately make predictions.
        stabilization_year: The year when asthma stabilization occurs.

    Returns:
        A DataFrame containing asthma occurrence predictions.
        Columns:

        * ``age (int)``: age in years, range ``[min_age, max_age]``.
        * ``sex (str)``: one of ``"M"`` or ``"F"``.
        * ``year (int)``: calendar year, range ``[starting_year, end_year]``.
        * ``incidence (float)``: predicted asthma incidence for the given age, sex, and year.
        * ``prevalence (float)``: predicted asthma prevalence for the given age, sex, and year.

    """
    df_asthma = pd.DataFrame(
        list(itertools.product(
            range(min_age, max_age + 1),
            ["F", "M"],
            range(starting_year, end_year + 1)
        )),
        columns=["age", "sex", "year"]
    )

    df_asthma["incidence"] = df_asthma.apply(
        lambda x: get_asthma_occurrence_prediction(
            x["age"], x["sex"], x["year"], "incidence", max_asthma_age, stabilization_year
        ),
        axis=1
    )
    df_asthma["prevalence"] = df_asthma.apply(
        lambda x: get_asthma_occurrence_prediction(
            x["age"], x["sex"], x["year"], "prevalence", max_asthma_age, stabilization_year
        ),
        axis=1
    )
    df_asthma["incidence"] = df_asthma.apply(
        lambda x: x["prevalence"] if x["age"] == 3 else x["incidence"],
        axis=1
    )
    return df_asthma


