import pandas as pd
import numpy as np
import datetime as dt
import itertools
from leap.utils import get_data_path, get_time_delta_tag, date_range, TimeDelta
from leap.logger import get_logger
from leap.data_generation.occurrence_calibration_data import get_asthma_occurrence_prediction
from leap.data_generation.utils import get_parser

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_TIMEPOINT = dt.datetime(1999, 1, 1)
STABILIZATION_TIMEPOINT = dt.datetime(2025, 1, 1)
MIN_ASTHMA_AGE = 3  # Minimum age for asthma diagnosis
MAX_ASTHMA_AGE = 62
MAX_AGE = 110
PROVINCES = ["BC", "CA"]
MAX_TIMEPOINTS = {
    "BC": dt.datetime(2043, 1, 1),
    "CA": dt.datetime(2066, 1, 1)
}


def get_asthma_df(
    time_delta: TimeDelta,
    min_timepoint: dt.datetime = MIN_TIMEPOINT,
    max_timepoint: dt.datetime = dt.datetime(2065, 1, 1),
    min_age: int = MIN_ASTHMA_AGE,
    max_age: int = MAX_AGE,
    max_asthma_age: int = MAX_ASTHMA_AGE,
    stabilization_timepoint: dt.datetime = STABILIZATION_TIMEPOINT
) -> pd.DataFrame:
    """Loads the asthma prevalence / incidence predictions from Model 1.

    Args:
        time_delta: The duration of time between two data points.
        min_timepoint: The starting timepoint for the dataframe.
        max_timepoint: The ending timepoint for the dataframe.
        min_age: The minimum age for asthma prediction.
        max_age: The maximum age for asthma prediction.
        max_asthma_age: The maximum age for for which the asthma prevalence / incidence
            model can accurately make predictions.
        stabilization_timepoint: The timepoint when asthma stabilization occurs.

    Returns:
        A DataFrame containing asthma occurrence predictions.
        Columns:

        * ``age (int)``: age in years, range ``[min_age, max_age]``.
        * ``sex (str)``: one of ``"M"`` or ``"F"``.
        * ``timepoint (datetime)``: timepoint, range ``[min_timepoint, max_timepoint]``.
        * ``incidence (float)``: predicted asthma incidence for the given age, sex, and timepoint.
        * ``prevalence (float)``: predicted asthma prevalence for the given age, sex, and timepoint.

    """
    df_asthma = pd.DataFrame(
        list(itertools.product(
            range(min_age, max_age + 1),
            ["F", "M"],
            list(date_range(min_timepoint, max_timepoint + time_delta, time_delta))
        )),
        columns=["age", "sex", "timepoint"]
    )

    df_asthma["incidence"] = df_asthma.apply(
        lambda x: get_asthma_occurrence_prediction(
            x["age"], x["sex"], x["timepoint"], "incidence", max_asthma_age, stabilization_timepoint
        ),
        axis=1
    )
    df_asthma["prevalence"] = df_asthma.apply(
        lambda x: get_asthma_occurrence_prediction(
            x["age"], x["sex"], x["timepoint"], "prevalence", max_asthma_age, stabilization_timepoint
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
        prevalence_past: The prevalence of asthma from the previous timepoint.
        prevalence_current: The prevalence of asthma in the current timepoint.
        incidence_current: The incidence of asthma in the current timepoint.

    Returns:
        The probability that someone diagnosed with asthma will maintain their diagnosis in the
        current timepoint.
    """

    prob = (prevalence_current - incidence_current * (1 - prevalence_past)) / prevalence_past
    return max(0, min(prob, 1))


def get_reassessment_data(
    df_asthma: pd.DataFrame,
    province: str = "CA",
    min_timepoint: dt.datetime = MIN_TIMEPOINT,
    max_timepoint: dt.datetime = dt.datetime(2065, 1, 1),
    max_age: int = MAX_AGE
) -> pd.DataFrame:
    """Generates reassessment data for asthma prevalence and incidence.

    Args:
        df_asthma: A dataframe containing asthma prevalence and incidence predictions from
            Occurrence Model 1. The dataframe should have the following columns:

            * ``age (int)``: age in years, range ``[3, max_age]``.
            * ``sex (str)``: one of ``"M"`` or ``"F"``.
            * ``timepoint (int)``: timepoint, range ``[min_timepoint, max_timepoint]``.
            * ``incidence (float)``: predicted asthma incidence for the given age, sex, and timepoint.
            * ``prevalence (float)``: predicted asthma prevalence for the given age, sex, and timepoint.

        province: The 2-letter province code, e.g. ``"CA"``.
        min_timepoint: The starting timepoint for the data.
        max_timepoint: The ending timepoint for the data.
        max_age: The maximum age for asthma prediction.

    Returns:
        A DataFrame containing the reassessment data.
        Columns:

        * ``timepoint (int)``: timepoint, range ``[min_timepoint + 1, max_timepoint]``.
        * ``province (str)``: the 2-letter province code, e.g. ``"CA"``.
        * ``age (int)``: age in years, range ``[4, max_age]``.
        * ``sex (str)``: one of ``"M"`` or ``"F"``.
        * ``prob (float)``: the probability that someone diagnosed with asthma will
          maintain their asthma diagnosis in the given timepoint. Range: ``[0, 1]``.
    """

    df_asthma_grouped = df_asthma.groupby("timepoint")

    df_reassessment = pd.DataFrame({
        "timepoint": np.array([], dtype=dt.datetime),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "prob": []
    })

    for timepoint in date_range(min_timepoint + time_delta, max_timepoint + time_delta, time_delta):

        # Get the predicted prevalence for the previous timepoint
        df_timepoint_0 = df_asthma_grouped.get_group(timepoint - 1)
        df_timepoint_0 = df_timepoint_0.loc[df_timepoint_0["age"] < max_age]
        df_timepoint_0["age_current"] = df_timepoint_0.apply(
             lambda x: x["age"] + 1,
                axis=1
        )
        df_timepoint_0.rename(columns={"age": "age_past", "timepoint": "timepoint_past"}, inplace=True)

        # Get the predicted prevalence for the current timepoint
        df_timepoint_1 = df_asthma_grouped.get_group(timepoint)
        df_timepoint_1 = df_timepoint_1.loc[df_timepoint_1["age"] > 3]
        df_timepoint_1.rename(columns={"age": "age_current", "timepoint": "timepoint_current"}, inplace=True)


        df = pd.merge(
            df_timepoint_0, df_timepoint_1, on=["age_current", "sex"], suffixes=("_past", "_current"), how="outer"
        )
        df["prob"] = df.apply(
            lambda x: calculate_reassessment_probability(
                x["prevalence_past"], x["prevalence_current"], x["incidence_current"]
            ),
            axis=1
        )

        df.drop(
            columns=[
                "prevalence_past", "prevalence_current", "incidence_current", "incidence_past",
                "age_past", "timepoint_past"
            ],
            inplace=True
        )
        df.rename(
            columns={"timepoint_current": "timepoint", "age_current": "age"}, inplace=True
        )
        df["province"] = [province] * df.shape[0]
        df_reassessment = pd.concat([df_reassessment, df], axis=0)
    
    return df_reassessment


def generate_reassessment_data(time_delta: TimeDelta):
    """Generate reassessment data for asthma prevalence and incidence across different provinces.
    
    Args:
        time_delta: The duration of time between two data points.
        
    """

    df_reassessment = pd.DataFrame({
        "timepoint": np.array([], dtype=dt.datetime),
        "province": [],
        "age": np.array([], dtype=int),
        "sex": [],
        "prob": []
    })

    for province in PROVINCES:
        df_asthma = get_asthma_df(
            time_delta=time_delta,
            min_timepoint=MIN_TIMEPOINT,
            max_timepoint=MAX_TIMEPOINTS[province],
            min_age=MIN_ASTHMA_AGE,
            max_age=MAX_AGE,
            max_asthma_age=MAX_ASTHMA_AGE,
            stabilization_timepoint=STABILIZATION_TIMEPOINT
        )
        df = get_reassessment_data(
            df_asthma=df_asthma,
            province=province,
            max_timepoint=MAX_TIMEPOINTS[province],
            max_age=MAX_AGE
        )
        df_reassessment = pd.concat([df_reassessment, df], axis=0)

    df_reassessment.reset_index(drop=True, inplace=True)

    time_delta_tag = get_time_delta_tag(time_delta)
    df_reassessment.to_csv(
        get_data_path(f"processed_data/{time_delta_tag}/asthma_reassessment.csv", mkdirs=True),
        index=False
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    time_delta = TimeDelta(iso_string=args.time_delta)
    generate_reassessment_data(time_delta=time_delta)