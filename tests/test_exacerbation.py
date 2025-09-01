import pytest
import pathlib
import json
from leap.exacerbation import Exacerbation
from leap.control import ControlLevels
from leap.utils import round_number
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config["exacerbation"]


@pytest.mark.parametrize(
    "hyperparameters, parameters, province",
    [
        (
            {
                "β0_μ": 0.0,
                "β0_σ": 0.0000001
            },
            {
                "βcontrol_C": -1.671,
                "βcontrol_PC": -0.9781,
                "βcontrol_UC": -0.5727,
                "min_year": 2001
            },
            "BC"
        ),
    ]
)
def test_exacerbation_constructor(config, hyperparameters, parameters, province):
    exacerbation = Exacerbation(config=config, province=province)
    assert exacerbation.hyperparameters["β0_μ"] == hyperparameters["β0_μ"]
    assert exacerbation.hyperparameters["β0_σ"] == hyperparameters["β0_σ"]
    assert round_number(
        exacerbation.parameters["βcontrol_C"], sigdigits=4
    ) == parameters["βcontrol_C"]
    assert round_number(
        exacerbation.parameters["βcontrol_PC"], sigdigits=4
    ) == parameters["βcontrol_PC"]
    assert round_number(
        exacerbation.parameters["βcontrol_UC"], sigdigits=4
    ) == parameters["βcontrol_UC"]
    assert exacerbation.parameters["β0"] < 50
    assert exacerbation.parameters["β0"] > -50
    assert exacerbation.parameters["min_year"] == parameters["min_year"]
    assert exacerbation.calibration_table.get_group(
        (parameters["min_year"] - 1, 0)
    )["age"].iloc[0] == 3


@pytest.mark.parametrize(
    "hyperparameters, control_levels, province, year, age, sex, lower_bound",
    [
        (
            {
                "β0_μ": 2.0,
                "β0_σ": 0.0000001
            },
            ControlLevels(fully_controlled=0.0, partially_controlled=0.0, uncontrolled=1.0),
            "BC",
            2001,
            4,
            False,
            1
        ),
    ]
)
def test_exacerbation_compute_num_exacerbations(
    config, hyperparameters, control_levels, province, year, age, sex, lower_bound
):
    config["hyperparameters"] = hyperparameters
    exacerbation = Exacerbation(config=config, province=province)
    num_exacerbations = exacerbation.compute_num_exacerbations(
        age=age, sex=sex, year=year, control_levels=control_levels
    )
    assert num_exacerbations > lower_bound
