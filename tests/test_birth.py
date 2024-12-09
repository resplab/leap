import pytest
import pandas as pd
from leap.birth import Birth
from leap.utils import round_number
from leap.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.parametrize(
    "starting_year, province, population_growth_type, expected_N, expected_prop_male",
    [
        (
            2022,
            "BC",
            "LG",
            42200,
            0.51185
        ),
    ]
)
def test_birth_constructor(
    starting_year, province, population_growth_type, expected_N, expected_prop_male
):
    birth = Birth(
        starting_year=starting_year,
        province=province,
        population_growth_type=population_growth_type
    )

    assert birth.estimate.get_group(starting_year)["N_relative"].iloc[0] == 1.0
    assert birth.estimate["province"].get_group(starting_year).iloc[0] == province
    assert birth.estimate.get_group(starting_year)["N"].iloc[0] == expected_N
    assert round_number(birth.estimate.get_group(starting_year)["prop_male"].iloc[0], sigdigits=5) == expected_prop_male
    assert birth.estimate.get_group(starting_year)["projection_scenario"].iloc[0] == population_growth_type
    assert birth.estimate.get_group(starting_year)["year"].iloc[0] == starting_year


@pytest.mark.parametrize(
    (
        "starting_year, province, population_growth_type, max_age, initial_population,"
        "expected_indices"
    ),
    [
        (
            2024,
            "BC",
            "M3",
            2,
            pd.DataFrame({"age": [0, 1, 2], "prop": [1.0, 2.0, 0.5]}),
            [0, 0, 1, 1, 1, 1, 2]
        ),
    ]
)
def test_birth_get_initial_population_indices(
    starting_year, province, population_growth_type, max_age, initial_population, expected_indices
):

    birth = Birth(
        starting_year=starting_year,
        province=province,
        population_growth_type=population_growth_type
    )
    birth.initial_population = initial_population
    initial_pop_indices = birth.get_initial_population_indices(max_age)
    assert initial_pop_indices == expected_indices


@pytest.mark.parametrize(
    (
        "starting_year, province, population_growth_type, max_age, year, num_births_initial,"
        "expected_num_newborn"
    ),
    [
        (
            2022,
            "BC",
            "LG",
            2,
            2024,
            1000,
            982
        ),
    ]
)
def test_birth_get_num_newborn(
    starting_year, province, population_growth_type, max_age, year, num_births_initial,
    expected_num_newborn
):

    birth = Birth(
        starting_year=starting_year,
        province=province,
        population_growth_type=population_growth_type
    )
    num_new_born = birth.get_num_newborn(num_births_initial, year)
    assert num_new_born == expected_num_newborn
