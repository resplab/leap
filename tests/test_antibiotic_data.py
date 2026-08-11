import pytest
import datetime as dt
import numpy as np
from leap.data_generation.antibiotic_data import convert_sex_to_numeric, load_birth_data


@pytest.mark.parametrize(
    "sex, expected_sex",
    [
        ("F", 1),
        ("M", 2),
    ]
)
def test_convert_sex_to_numeric(sex, expected_sex):
    assert convert_sex_to_numeric(sex) == expected_sex


def test_load_birth_data():
    df_birth = load_birth_data()
    assert df_birth.shape[0] > 0
    assert df_birth.columns.tolist() == ["timepoint", "province", "sex", "n_birth"]
