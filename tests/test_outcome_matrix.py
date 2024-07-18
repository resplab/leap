import pytest
import pathlib
import json
import numpy as np
import pandas as pd
from leap.outcome_matrix import OutcomeMatrix, OutcomeTable
from tests.utils import __test_dir__
from leap.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.parametrize(
    "until_all_die, min_year, max_year, max_age",
    [
        (
            False,
            2001,
            2041,
            111
        ),
    ]
)
def test_outcome_matrix_constructor(
    until_all_die, min_year, max_year, max_age
):

    outcome_matrix = OutcomeMatrix(
        until_all_die=until_all_die,
        min_year=min_year,
        max_year=max_year,
        max_age=max_age
    )
    time_horizon = max_year - min_year + 1
    assert outcome_matrix.min_year == min_year
    assert outcome_matrix.max_year == max_year
    assert outcome_matrix.max_age == max_age

    alive_df = outcome_matrix.alive.grouped_data.get_group((min_year))
    assert alive_df.shape == ((max_age + 1) * 2, 4)
    control_df = outcome_matrix.control.grouped_data.get_group((min_year, 1))
    assert control_df.shape == ((max_age + 1) * 2, 5)
    assert len(outcome_matrix.control.grouped_data.groups) == time_horizon * 3
    exacerbation_by_severity_df = outcome_matrix.exacerbation_by_severity.grouped_data.get_group(
        (min_year, 3)
    )
    assert exacerbation_by_severity_df.shape == ((max_age + 1) * 2, 5)
    assert isinstance(outcome_matrix.asthma_prevalence_contingency_table, OutcomeTable)


@pytest.mark.parametrize(
    "until_all_die, min_year, max_year, max_age, age, sex, increment",
    [
        (
            False,
            2001,
            2041,
            111,
            4,
            "M",
            3
        ),
    ]
)
def test_outcome_matrix_increment(
    until_all_die, min_year, max_year, max_age, age, sex, increment
):

    outcome_matrix = OutcomeMatrix(
        until_all_die=until_all_die,
        min_year=min_year,
        max_year=max_year,
        max_age=max_age
    )
    outcome_matrix.antibiotic_exposure.increment(
        column="n_antibiotic_exposure",
        filter_columns={
            "year": min_year,
            "age": age,
            "sex": sex
        },
        amount=increment
    )
    df = outcome_matrix.antibiotic_exposure.data
    df = df[(df["age"] == age) & (df["sex"] == sex) & (df["year"] == min_year)]
    assert df["n_antibiotic_exposure"].values[0] == increment
