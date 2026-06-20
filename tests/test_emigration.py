import pytest
import datetime as dt
from leap.emigration import Emigration
from leap.utils import round_number


@pytest.mark.parametrize(
    "min_timepoint, timepoint, age, sex, province, projection_scenario, value",
    [
        (
            dt.datetime(2024, 1, 1),
            dt.datetime(2025, 1, 1),
            89,
            "F",
            "BC",
            "LG",
            0.02836
        ),
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2001, 1, 1),
            5,
            "M",
            "BC",
            "past",
            0.001127
        ),
    ]
)
def test_emigration_constructor(
    min_timepoint, timepoint, age, sex, province, projection_scenario, value
):
    emigration = Emigration(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario
    )

    df = emigration.table.get_group((timepoint))
    assert round_number(
        df[(df["age"] == age) & (df["sex"] == sex)]["prob_emigration"].values[0],
        sigdigits=4
    ) == value



@pytest.mark.parametrize(
    "min_timepoint, timepoint, age, sex, province, projection_scenario, lower_bound, upper_bound",
    [
        (
            dt.datetime(2020, 1, 1),
            dt.datetime(2023, 1, 1),
            0,
            "F",
            "BC",
            "FA",
            0,
            0
        ),
        (
            dt.datetime(2020, 1, 1),
            dt.datetime(2023, 1, 1),
            99,
            "M",
            "BC",
            "M2",
            42400,
            43800
        ),
    ]
)
def test_emigration_compute_probability(
    min_timepoint, timepoint, age, sex, province, projection_scenario, lower_bound, upper_bound
):
    """Test the ``compute_probability`` method of the ``Emigration`` class.
    
    The ``compute_probability`` method should return a boolean value indicating
    whether the agent emigrates in a given timepoint. This is computed using the binomial distribution.
    For the purposes of testing, we will run the method 100,000 times and check that the number of
    emigrants falls within a certain range. Please see
    ``processed_data/migration/migration_table.csv`` for the corresponding values.
    """

    emigration = Emigration(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario
    )

    count = 0
    for _ in range(100000):
        if emigration.compute_probability(timepoint, age, sex):
            count += 1

    assert round_number(count, sigdigits=3) <= upper_bound
    assert round_number(count, sigdigits=3) >= lower_bound
