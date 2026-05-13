import pytest
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from leap.outcome_matrix import OutcomeMatrix, OutcomeTable, combine_outcome_matrices, \
    combine_outcome_tables
from leap.utils import date_range
from leap.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.parametrize(
    "until_all_die, min_timepoint, max_timepoint, max_age, time_delta",
    [
        (
            False,
            dt.datetime(2001, 1, 1),
            dt.datetime(2041, 1, 1),
            111,
            relativedelta(years=1)
        ),
    ]
)
def test_outcome_matrix_constructor(
    until_all_die, min_timepoint, max_timepoint, max_age, time_delta
):

    outcome_matrix = OutcomeMatrix(
        until_all_die=until_all_die,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
        max_age=max_age,
        time_delta=time_delta
    )
    n_timepoints = len(list(date_range(min_timepoint, max_timepoint + time_delta, time_delta)))
    assert outcome_matrix.min_timepoint == min_timepoint
    assert outcome_matrix.max_timepoint == max_timepoint
    assert outcome_matrix.max_age == max_age

    alive_df = outcome_matrix.alive.data
    assert alive_df.shape == ((max_age + 1) * 2 * n_timepoints, 4)
    control_df = outcome_matrix.control.grouped_data.get_group((min_timepoint, 1))
    assert control_df.shape == ((max_age + 1) * 2, 5)
    assert len(outcome_matrix.control.grouped_data.groups) == n_timepoints * 3
    exacerbation_by_severity_df = outcome_matrix.exacerbation_by_severity.grouped_data.get_group(
        (min_timepoint, 3)
    )
    assert exacerbation_by_severity_df.shape == ((max_age + 1) * 2, 5)
    assert isinstance(outcome_matrix.asthma_prevalence_contingency_table, OutcomeTable)


@pytest.mark.parametrize(
    "until_all_die, min_timepoint, max_timepoint, time_delta, max_age, age, sex, increment",
    [
        (
            False,
            dt.datetime(2001, 1, 1),
            dt.datetime(2041, 1, 1),
            relativedelta(years=1),
            111,
            4,
            "M",
            3
        ),
    ]
)
def test_outcome_matrix_increment(
    until_all_die, min_timepoint, max_timepoint, time_delta, max_age, age, sex, increment
):

    outcome_matrix = OutcomeMatrix(
        until_all_die=until_all_die,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
        max_age=max_age,
        time_delta=time_delta
    )
    outcome_matrix.antibiotic_exposure.increment(
        column="n_antibiotic_exposure",
        filter_columns={
            "timepoint": min_timepoint,
            "age": age,
            "sex": sex
        },
        amount=increment
    )
    df = outcome_matrix.antibiotic_exposure.data
    df = df[(df["age"] == age) & (df["sex"] == sex) & (df["timepoint"] == min_timepoint)]
    assert df["n_antibiotic_exposure"].values[0] == increment


@pytest.mark.parametrize(
    "outcome_data, expected_outcome_data, column",
    [
        (
            [
                pd.DataFrame({
                    "timepoint": [dt.datetime(2001, 1, 1), dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1), dt.datetime(2002, 1, 1)],
                    "sex": ["M", "F", "M", "F"],
                    "age": [4, 4, 4, 4],
                    "n": [3, 5, 2, 4]
                }),
                pd.DataFrame({
                    "timepoint": [dt.datetime(2001, 1, 1), dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1), dt.datetime(2002, 1, 1)],
                    "sex": ["M", "F", "M", "F"],
                    "age": [4, 4, 4, 4],
                    "n": [5, 7, 3, 1]
                }),
                pd.DataFrame({
                    "timepoint": [dt.datetime(2001, 1, 1), dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1), dt.datetime(2002, 1, 1)],
                    "sex": ["M", "F", "M", "F"],
                    "age": [4, 4, 4, 4],
                    "n": [6, 0, 1, 3]
                })
            ],
            pd.DataFrame({
                "timepoint": [dt.datetime(2001, 1, 1), dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1), dt.datetime(2002, 1, 1)],
                "sex": ["M", "F", "M", "F"],
                "age": [4, 4, 4, 4],
                "n": [14, 12, 6, 8]
            }),
            "n"
        ),
    ]
)
def test_combine_outcome_tables(
    outcome_data, expected_outcome_data, column
):

    combined_outcome_table = combine_outcome_tables(
        outcome_tables=[OutcomeTable(data=df) for df in outcome_data],
        column=column
    )
    pd.testing.assert_frame_equal(
        combined_outcome_table.data.reset_index(drop=True),
        expected_outcome_data.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "until_all_die, min_timepoint, max_timepoint, time_delta, max_age",
    [
        (
            False,
            dt.datetime(2001, 1, 1),
            dt.datetime(2041, 1, 1),
            relativedelta(years=1),
            111
        ),
    ]
)
def test_combine_outcome_matrices(
    until_all_die, min_timepoint, max_timepoint, time_delta, max_age
):

    outcome_matrix1 = OutcomeMatrix(
        until_all_die=until_all_die,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
        time_delta=time_delta,
        max_age=max_age
    )
    outcome_matrix2 = OutcomeMatrix(
        until_all_die=until_all_die,
        min_timepoint=min_timepoint,
        max_timepoint=max_timepoint,
        time_delta=time_delta,
        max_age=max_age
    )

    outcome_matrix1.antibiotic_exposure.increment(
        column="n_antibiotic_exposure",
        filter_columns={
            "timepoint": min_timepoint,
            "age": 4,
            "sex": "M"
        },
        amount=3
    )
    outcome_matrix2.antibiotic_exposure.increment(
        column="n_antibiotic_exposure",
        filter_columns={
            "timepoint": min_timepoint,
            "age": 4,
            "sex": "M"
        },
        amount=5
    )
    combined_outcome_matrix = combine_outcome_matrices(
        [outcome_matrix1, outcome_matrix2]
    )
    df = combined_outcome_matrix.antibiotic_exposure.data
    df = df[(df["age"] == 4) & (df["sex"] == "M") & (df["timepoint"] == min_timepoint)]
    assert df["n_antibiotic_exposure"].values[0] == 8