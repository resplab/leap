import pytest
import datetime as dt
import numpy as np
from leap.data_generation.antibiotic_data import convert_sex_to_numeric, \
    convert_timepoint_to_numeric, load_birth_data


@pytest.mark.parametrize(
    "sex, expected_sex",
    [
        ("F", 1),
        ("M", 2),
    ]
)
def test_convert_sex_to_numeric(sex, expected_sex):
    assert convert_sex_to_numeric(sex) == expected_sex


@pytest.mark.parametrize(
    "timepoint, expected_timepoint",
    [
        (dt.datetime(2000, 1, 1), 2000.0),
        (dt.datetime(2000, 2, 1), 2000.0821917808219),
    ]
)
def test_convert_timepoint_to_numeric(timepoint, expected_timepoint):
    np.testing.assert_almost_equal(
        convert_timepoint_to_numeric(timepoint),
        expected_timepoint,
        decimal=5
    )


def test_load_birth_data():
    df_birth = load_birth_data()
    assert df_birth.shape[0] > 0
    assert df_birth.columns.tolist() == ["timepoint", "province", "sex", "n_birth"]
