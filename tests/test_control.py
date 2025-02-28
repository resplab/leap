import pytest
import pathlib
import json
import numpy as np
from leap.control import Control
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config["control"]


@pytest.mark.parametrize(
    "hyperparameters, parameters",
    [
        (
            {
                "β0_μ": 0.0,
                "β0_σ": 1.678728
            },
            {
                "βage": 3.5430381,
                "βage2": -3.4980710,
                "βsexage": -0.8161495,
                "βsexage2": -1.1654264,
                "βsex": 0.2347807,
                "θ": [-0.3950, 2.754],
                "β0": 0.0
            }
        ),
    ]
)
def test_control_constructor(
    config, hyperparameters, parameters
):
    control = Control(config=config)
    assert control.hyperparameters["β0_μ"] == hyperparameters["β0_μ"]
    assert control.hyperparameters["β0_σ"] == hyperparameters["β0_σ"]
    assert control.parameters["βage"] == parameters["βage"]
    assert control.parameters["βage2"] == parameters["βage2"]
    assert control.parameters["βsexage"] == parameters["βsexage"]
    assert control.parameters["βsexage2"] == parameters["βsexage2"]
    assert control.parameters["βsex"] == parameters["βsex"]
    np.testing.assert_array_equal(control.parameters["θ"], parameters["θ"])
    assert control.parameters["β0"] > -50
    assert control.parameters["β0"] < 50


@pytest.mark.parametrize(
    "sex, age, initial",
    [
        (
            False,
            20,
            False
        ),
    ]
)
def test_control_compute_control_levels(config, sex, age, initial):
    control = Control(config=config)
    control_levels = control.compute_control_levels(sex=sex, age=age, initial=initial)
    assert round(sum([
        control_levels.fully_controlled,
        control_levels.partially_controlled,
        control_levels.uncontrolled
        ]), 0) == 1
    assert round(sum(control_levels.as_array()), 0) == 1


