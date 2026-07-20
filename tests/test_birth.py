import pytest
import pandas as pd
import datetime as dt
from leap.birth import Birth
from leap.utils import round_number
from leap.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.parametrize(
    "min_timepoint, province, projection_scenario, expected_N, expected_prop_male",
    [
        (
            dt.datetime(2022, 1, 1),
            "BC",
            "LG",
            42300,
            0.51300
        ),
    ]
)
def test_birth_constructor(
    min_timepoint, province, projection_scenario, expected_N, expected_prop_male
):
    birth = Birth(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario
    )

    assert birth.estimate.get_group(min_timepoint)["N_relative"].iloc[0] == 1.0
    assert birth.estimate["province"].get_group(min_timepoint).iloc[0] == province
    assert birth.estimate.get_group(min_timepoint)["N"].iloc[0] == expected_N
    assert round_number(birth.estimate.get_group(min_timepoint)["prop_male"].iloc[0], sigdigits=5) == expected_prop_male
    assert birth.estimate.get_group(min_timepoint)["projection_scenario"].iloc[0] == projection_scenario
    assert birth.estimate.get_group(min_timepoint)["timepoint"].iloc[0] == min_timepoint


@pytest.mark.parametrize(
    (
        "min_timepoint, province, projection_scenario, max_age, initial_population,"
        "expected_indices"
    ),
    [
        (
            dt.datetime(2024, 1, 1),
            "BC",
            "M3",
            2,
            pd.DataFrame({"age": [0, 1, 2], "prop": [1.0, 2.0, 0.5]}),
            [0, 0, 1, 1, 1, 1, 2]
        ),
    ]
)
def test_birth_get_initial_population_indices(
    min_timepoint, province, projection_scenario, max_age, initial_population, expected_indices
):

    birth = Birth(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario
    )
    birth.initial_population = initial_population
    initial_pop_indices = birth.get_initial_population_indices(max_age)
    assert initial_pop_indices == expected_indices


@pytest.mark.parametrize(
    (
        "min_timepoint, province, projection_scenario, max_age, timepoint, num_births_initial,"
        "expected_num_newborn"
    ),
    [
        (
            dt.datetime(2022, 1, 1),
            "BC",
            "LG",
            2,
            dt.datetime(2024, 1, 1),
            1000,
            993
        ),
    ]
)
def test_birth_get_num_newborn(
    min_timepoint, province, projection_scenario, max_age, timepoint, num_births_initial,
    expected_num_newborn
):

    birth = Birth(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario
    )
    num_new_born = birth.get_num_newborn(num_births_initial, timepoint)
    assert num_new_born == expected_num_newborn
