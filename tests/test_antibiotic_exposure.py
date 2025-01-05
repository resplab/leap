import pytest
import pathlib
import json
from leap.antibiotic_exposure import AntibioticExposure
from leap.utils import round_number
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config["antibiotic_exposure"]


def test_antibiotic_exposure_constructor(config):
    antibiotic_exposure = AntibioticExposure(config=config)
    assert antibiotic_exposure.parameters["β0"] == 110.000442
    assert antibiotic_exposure.mid_trends.get_group((2002, 0))["year"].iloc[0] == 2002


@pytest.mark.parametrize(
    "parameters, year, sex, expected_probability",
    [
        (
            {
                "β0": 1,
                "βyear": 0.01,
                "βsex": 1,
                "θ": 500,
                "fixyear": None,
                "βfloor": 0.05,
                "β2005": 1,
                "β2005_year": 1
            },
            2001,
            False,
            0.000000375
        ),
    ]
)
def test_antibiotic_exposure_compute_probability(parameters, year, sex, expected_probability):
    antibiotic_exposure = AntibioticExposure(parameters=parameters)
    probability = antibiotic_exposure.compute_probability(sex, year)
    assert round_number(probability, sigdigits=3) == expected_probability


@pytest.mark.parametrize(
    "parameters, birth_year, sex, expected_probability",
    [
        (
            {
                "β0": -100000,
                "βyear": -0.01,
                "βsex": -1,
                "θ": 500,
                "fixyear": None,
                "βfloor": 0.0,
                "β2005": 1,
                "β2005_year": 1
            },
            2001,
            False,
            0
        ),
    ]
)
def test_antibiotic_exposure_compute_num_antibiotic_use(
    parameters, birth_year, sex, expected_probability
):
    antibiotic_exposure = AntibioticExposure(parameters=parameters)
    num_antibiotic_use = antibiotic_exposure.compute_num_antibiotic_use(
        sex=sex, birth_year=birth_year
    )
    assert round_number(num_antibiotic_use, digits=1) == expected_probability
