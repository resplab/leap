import pytest
from leap.immigration import Immigration
from leap.utils import round_number


@pytest.mark.parametrize(
    (
        "starting_year, year, age, max_age, sex, province, population_growth_type,"
        "prop_immigrants_birth, prop_immigrants_year"
    ),
    [
        (
            2024,
            2026,
            4,
            111,
            0,
            "BC",
            "LG",
            0.004980,
            0.004936
        ),
        (
            2024,
            2025,
            4,
            111,
            1,
            "BC",
            "LG",
            0.007383,
            0.007294
        ),
    ]
)
def test_immigration_constructor(
    starting_year, year, age, max_age, sex, province, population_growth_type,
    prop_immigrants_birth, prop_immigrants_year
):
    immigration = Immigration(
        starting_year=starting_year,
        province=province,
        population_growth_type=population_growth_type,
        max_age=max_age
    )
    df = immigration.table.get_group((year))
    row = df[(df["age"] == age) & (df["sex"] == sex)]
    assert round_number(row["prop_immigrants_birth"].values[0], sigdigits=4) == prop_immigrants_birth
    assert round_number(row["prop_immigrants_year"].values[0], sigdigits=4) == prop_immigrants_year


@pytest.mark.parametrize(
    (
        "starting_year, year, max_age, province, population_growth_type,"
        "num_new_born, num_new_immigrants"
    ),
    [
        (
            2024,
            2025,
            111,
            "BC",
            "LG",
            1000,
            1013
        )
    ]
)
def test_immigration_get_num_new_immigrants(
    starting_year, year, max_age, province, population_growth_type, num_new_born,
    num_new_immigrants
):
    immigration = Immigration(
        starting_year=starting_year,
        province=province,
        population_growth_type=population_growth_type,
        max_age=max_age
    )
    assert immigration.get_num_new_immigrants(num_new_born, year) == num_new_immigrants
