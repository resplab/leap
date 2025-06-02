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
PROVINCES = ["BC", "CA"]
MAX_YEARS = {
    "BC": 2043,
    "CA": 2066
}


def get_asthma_df(
    starting_year: int = STARTING_YEAR,
    max_year: int = 2065,
    min_age: int = MIN_ASTHMA_AGE,
    max_age: int = MAX_AGE,
    max_asthma_age: int = MAX_ASTHMA_AGE,
    stabilization_year: int = STABILIZATION_YEAR
) -> pd.DataFrame:
    """Loads the asthma prevalence / incidence predictions from Model 1.

    Args:
        starting_year: The starting year for the dataframe.
        max_year: The ending year for the dataframe.
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
        * ``year (int)``: calendar year, range ``[starting_year, max_year]``.
        * ``incidence (float)``: predicted asthma incidence for the given age, sex, and year.
        * ``prevalence (float)``: predicted asthma prevalence for the given age, sex, and year.

    """
    df_asthma = pd.DataFrame(
        list(itertools.product(
            range(min_age, max_age + 1),
            ["F", "M"],
            range(starting_year, max_year + 1)
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


def calculate_reassessment_probability(
    prevalence_past: float,
    prevalence_current: float,
    incidence_current: float
) -> float:
    """Calculates the reassessment probability based on asthma prevalence and incidence.

    Args:
        prevalence_past: The prevalence of asthma from the previous year.
        prevalence_current: The prevalence of asthma in the current year.
        incidence_current: The incidence of asthma in the current year.

    Returns:
        The probability that someone diagnosed with asthma will maintain their diagnosis in the
        current year.
    """

    prob = (prevalence_current - incidence_current * (1 - prevalence_past)) / prevalence_past
    return max(0, min(prob, 1))


def get_reassessment_data(
    df_asthma: pd.DataFrame,
    province: str = "CA",
    starting_year: int = STARTING_YEAR,
    max_year: int = 2065,
    max_age: int = MAX_AGE
) -> pd.DataFrame:
    """Generates reassessment data for asthma prevalence and incidence.

    Args:
        df_asthma: A dataframe containing asthma prevalence and incidence predictions from
            Occurrence Model 1. The dataframe should have the following columns:

            * ``age (int)``: age in years, range ``[3, max_age]``.
            * ``sex (str)``: one of ``"M"`` or ``"F"``.
            * ``year (int)``: calendar year, range ``[starting_year, max_year]``.
            * ``incidence (float)``: predicted asthma incidence for the given age, sex, and year.
            * ``prevalence (float)``: predicted asthma prevalence for the given age, sex, and year.

        province: The 2-letter province code, e.g. ``"CA"``.
        starting_year: The starting year for the data.
        max_year: The ending year for the data.
        max_age: The maximum age for asthma prediction.

    Returns:
        A DataFrame containing the reassessment data.
        Columns:

        * ``year (int)``: calendar year, range ``[starting_year + 1, max_year]``.
        * ``province (str)``: the 2-letter province code, e.g. ``"CA"``.
        * ``age (int)``: age in years, range ``[4, max_age]``.
        * ``sex (str)``: one of ``"M"`` or ``"F"``.
        * ``reassessment (float)``: the probability that someone diagnosed with asthma will
          maintain their asthma diagnosis in the given year. Range: ``[0, 1]``.
    """

    df_asthma_grouped = df_asthma.groupby(["year"])

    df_reassessment = pd.DataFrame({
        "year": np.array([], dtype=int),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "reassessment": []
    })

    for year in range(starting_year + 1, max_year + 1):

        # Get the predicted prevalence for the previous year
        df_year_0 = df_asthma_grouped.get_group((year - 1,))
        df_year_0 = df_year_0.loc[df_year_0["age"] < max_age]
        df_year_0["age_current"] = df_year_0.apply(
             lambda x: x["age"] + 1,
                axis=1
        )
        df_year_0.rename(columns={"age": "age_past", "year": "year_past"}, inplace=True)

        # Get the predicted prevalence for the current year
        df_year_1 = df_asthma_grouped.get_group((year,))
        df_year_1 = df_year_1.loc[df_year_1["age"] > 3]
        df_year_1.rename(columns={"age": "age_current", "year": "year_current"}, inplace=True)


        df = pd.merge(
            df_year_0, df_year_1, on=["age_current", "sex"], suffixes=("_past", "_current"), how="outer"
        )
        df["reassessment"] = df.apply(
            lambda x: calculate_reassessment_probability(
                x["prevalence_past"], x["prevalence_current"], x["incidence_current"]
            ),
            axis=1
        )

        df.drop(
            columns=[
                "prevalence_past", "prevalence_current", "incidence_current", "incidence_past",
                "age_past", "year_past"
            ],
            inplace=True
        )
        df.rename(
            columns={"year_current": "year", "age_current": "age"}, inplace=True
        )
        df["province"] = [province] * df.shape[0]
        df_reassessment = pd.concat([df_reassessment, df], axis=0)
    
    return df_reassessment


def generate_reassessment_data():
    """Generate reassessment data for asthma prevalence and incidence across different provinces."""

    df_reassessment = pd.DataFrame({
        "year": np.array([], dtype=int),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "reassessment": []
    })

    for province in PROVINCES:
        df_asthma = get_asthma_df(
            starting_year=STARTING_YEAR,
            max_year=MAX_YEARS[province],
            min_age=MIN_ASTHMA_AGE,
            max_age=MAX_AGE,
            max_asthma_age=MAX_ASTHMA_AGE,
            stabilization_year=STABILIZATION_YEAR
        )
        df = get_reassessment_data(
            df_asthma=df_asthma,
            province=province,
            max_year=MAX_YEARS[province],
            max_age=MAX_AGE
        )
        df_reassessment = pd.concat([df_reassessment, df], axis=0)

    df_reassessment.reset_index(drop=True, inplace=True)
    df_reassessment.to_csv(get_data_path("processed_data/asthma_reassessment.csv"), index=False)


if __name__ == "__main__":
    generate_reassessment_data()