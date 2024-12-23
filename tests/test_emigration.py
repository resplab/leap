import pytest
from leap.emigration import Emigration
from leap.utils import round_number


@pytest.mark.parametrize(
    "starting_year, year, age, sex, province, population_growth_type, value",
    [
        (
            2024,
            2025,
            89,
            "F",
            "BC",
            "LG",
            0.02836
        ),
    ]
)
def test_emigration_constructor(
    starting_year, year, age, sex, province, population_growth_type, value
):
    emigration = Emigration(
        starting_year=starting_year,
        province=province,
        population_growth_type=population_growth_type
    )
    df = emigration.table.get_group((year))
    assert df["F"].iloc[0] == 0.0
    assert df["M"].iloc[0] == 0.0
    assert round_number(df[df["age"] == age][sex].values[0], sigdigits=4) == value


@pytest.mark.parametrize(
    "starting_year, year, age, sex, province, population_growth_type, lower_bound, upper_bound",
    [
        (
            2020,
            2023,
            0,
            "F",
            "BC",
            "FA",
            0,
            0
        ),
        (
            2020,
            2023,
            99,
            "M",
            "BC",
            "M2",
            43300,
            43600
        ),
    ]
)
def test_emigration_compute_probability(
    starting_year, year, age, sex, province, population_growth_type, lower_bound, upper_bound
):
    emigration = Emigration(
        starting_year=starting_year,
        province=province,
        population_growth_type=population_growth_type
    )

    count = 0
    for i in range(100000):
        if emigration.compute_probability(year, age, sex):
            count += 1

    assert round_number(count, sigdigits=3) <= upper_bound
    assert round_number(count, sigdigits=3) >= lower_bound
