import pathlib
import pandas as pd
import numpy as np
import itertools
from leap.utils import get_data_path
from leap.logger import get_logger
from typing import Tuple

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

PROVINCE = "CA"
MAX_YEAR = 2065 # 2065 for CA; 2043 for BC
MIN_YEAR = 2000
STABILIZATION_YEAR = 2025
BASELINE_YEAR = 2001
MAX_AGE = 63
MAX_AGE = 9
MAX_ASTHMA_AGE = 62
MIN_ASTHMA_AGE = 3
# odds ratio between asthma prevalence at age 3 and family history (CHILD Study)
OR_ASTHMA_AGE_3 = 1.13
# odds ratio between asthma prevalence at age 5 and family history (CHILD Study)
OR_ASTHMA_AGE_5 = 2.4
# beta parameter for the antibiotic dose term in the odds ratio equation for antibiotic courses
BETA_ABX_DOSE = 0.053
# beta parameter for the age term in the odds ratio equation for antibiotic courses
BETA_ABX_AGE = -0.225
# beta parameter for the constant term in the odds ratio equation for antibiotic courses
BETA_ABX_0 = 1.711 + 0.115
INC_BETA_PARAMS = [(np.log(OR_ASTHMA_AGE_5) - np.log(OR_ASTHMA_AGE_3)) / 2, BETA_ABX_AGE]
# the probability that one or more parents have asthma (CHILD Study)
PROB_FAM_HIST = 0.2927242

DF_OCC_PRED = pd.read_csv(get_data_path("processed_data/asthma_occurrence_predictions.csv"))


def asthma_predictor(age: int, sex: str, year: int, occurrence_type: str) -> float:
    """
    Predicts the asthma prevalence or incidence based on the given parameters.
    Args:
        age: Age of the individual.
        sex: One of "M" or "F".
        year: Year of the prediction.
        occurrence_type: One of "prevalence" or "incidence".
    Returns:
        A float representing the predicted asthma prevalence or incidence.
    """

    age = min(age, MAX_ASTHMA_AGE)
    year = min(year, STABILIZATION_YEAR)

    return DF_OCC_PRED.loc[
        (DF_OCC_PRED["age"] == age) &
        (DF_OCC_PRED["year"] == year) &
        (DF_OCC_PRED["sex"] == sex)
    ][occurrence_type].values[0]


def load_occurrence_data(
    province: str = PROVINCE,
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the asthma incidence and prevalence data for the given province and year range.

    Args:
        province: The province to load data for.
        min_year: The minimum year to load data for.
        max_year: The maximum year to load data for.

    Returns:
        A tuple of two DataFrames: the first is the incidence data, and the second is the
        prevalence data.
    """

    df_asthma = pd.DataFrame(
        list(itertools.product(range(3, 111), [0, 1], range(min_year, max_year + 1))),
        columns=["age", "sex", "year"]
    )

    df_asthma["incidence"] = df_asthma.apply(
        lambda x: asthma_predictor(x["age"], x["sex"], x["year"], "inc"),
        axis=1
    )
    df_asthma["prevalence"] = df_asthma.apply(
        lambda x: asthma_predictor(x["age"], x["sex"], x["year"], "prev"),
        axis=1
    )
    df_asthma["incidence"] = df_asthma.apply(
        lambda x: x["prevalence"] if x["age"] == 3 else x["incidence"],
        axis=1
    )
    df_asthma["province"] = [province] * df_asthma.shape[0]

    df_incidence = df_asthma[["year", "age", "sex", "incidence"]].copy()
    df_prevalence = df_asthma[["year", "age", "sex", "prevalence"]].copy()

    return df_incidence, df_prevalence


def load_reassessment_data(
    province: str = PROVINCE
) -> pd.DataFrame:
    """Load the asthma reassessment data for the given province.

    Args:
        province: The province to load data for.

    Returns:
        A DataFrame containing the reassessment data.
    """

    df_reassessment = pd.read_csv(get_data_path("processed_data/asthma_reassessment.csv"))
    df_reassessment = df_reassessment.loc[df_reassessment["province"] == province]

    df_reassessment = df_reassessment.drop(columns=["province"])

    return df_reassessment



    

