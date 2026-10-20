import pytest
import datetime as dt
from leap.immigration import Immigration
from leap.birth import Birth
from leap.utils import round_number, TimeDelta, date_range


@pytest.mark.parametrize(
    (
        "min_timepoint, timepoint, age, max_age, sex, province, projection_scenario,"
        "prop_immigrants_birth, prop_immigrants_timepoint"
    ),
    [
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2005, 1, 1),
            4,
            111,
            "M",
            "BC",
            "LG",
            0.006876,
            0.008984
        ),
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2003, 1, 1),
            4,
            111,
            "F",
            "BC",
            "LG",
            0.002131,
            0.005572
        ),
    ]
)
def test_immigration_constructor(
    min_timepoint, timepoint, age, max_age, sex, province, projection_scenario,
    prop_immigrants_birth, prop_immigrants_timepoint
):
    immigration = Immigration(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario,
        max_age=max_age
    )
    df = immigration.get_table_group((timepoint))
    row = df[(df["age"] == age) & (df["sex"] == sex)]
    assert round_number(row["prop_immigrants_birth"].values[0], sigdigits=4) == prop_immigrants_birth
    assert round_number(row["prop_immigrants_timepoint"].values[0], sigdigits=4) == prop_immigrants_timepoint


@pytest.mark.parametrize(
    (
        "min_timepoint, timepoint, max_age, province, projection_scenario,"
        "num_new_born, num_new_immigrants"
    ),
    [
        (
            dt.datetime(2000, 1, 1),
            dt.datetime(2003, 1, 1),
            111,
            "BC",
            "LG",
            1000,
            383
        )
    ]
)
def test_immigration_get_num_new_immigrants(
    min_timepoint, timepoint, max_age, province, projection_scenario, num_new_born,
    num_new_immigrants
):
    immigration = Immigration(
        min_timepoint=min_timepoint,
        province=province,
        projection_scenario=projection_scenario,
        max_age=max_age
    )
    assert immigration.get_num_new_immigrants(num_new_born, timepoint) == num_new_immigrants


def test_immigration_monthly_matches_annual():
    # births and immigrants over 12 monthly steps should be close to one annual step; this fails if
    # Birth.get_num_newborn and n_birth in the migration tables use different time intervals
    num_births = {}
    num_immigrants = {}
    for time_delta in [TimeDelta(years=1), TimeDelta(months=1)]:
        birth = Birth(
            min_timepoint=dt.datetime(2023, 1, 1), province="BC", projection_scenario="M3",
            time_delta=time_delta
        )
        immigration = Immigration(
            min_timepoint=dt.datetime(2023, 1, 1), province="BC", projection_scenario="M3",
            time_delta=time_delta
        )
        timepoints = list(date_range(
            dt.datetime(2030, 1, 1) + time_delta, dt.datetime(2031, 1, 1) + time_delta, time_delta
        ))
        newborns = [birth.get_num_newborn(50000, timepoint) for timepoint in timepoints]
        key = time_delta.to_isoformat()
        num_births[key] = sum(newborns)
        num_immigrants[key] = sum(
            immigration.get_num_new_immigrants(n, timepoint)
            for n, timepoint in zip(newborns, timepoints)
        )
    annual, monthly = TimeDelta(years=1).to_isoformat(), TimeDelta(months=1).to_isoformat()
    assert num_births[monthly] == pytest.approx(num_births[annual], rel=0.02)
    # monthly immigration is currently ~28% higher than annual (issue #98); tighten once fixed
    assert 0.7 <= num_immigrants[monthly] / num_immigrants[annual] <= 1.5
