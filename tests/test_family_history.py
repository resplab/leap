import pytest
from leap.family_history import FamilyHistory


@pytest.mark.parametrize(
    "parameters, is_valid",
    [
        (
            {
                "p": 0.0,
            },
            True
        ),
        (
            {
                "p": 1.0,
            },
            True
        ),
        (
            {
                "p": 0.5,
            },
            True
        ),
        (
            {
                "p": 5.6,
            },
            False
        ),
    ]
)
def test_family_history_constructor(parameters, is_valid):
    if is_valid:
        family_history = FamilyHistory(probability=parameters["p"])
        assert family_history.probability == parameters["p"]
    else:
        with pytest.raises(ValueError) as error:
            family_history = FamilyHistory(probability=parameters["p"])
        assert str(error.value) == "p must be a probability between 0 and 1, received 5.6."


@pytest.mark.parametrize(
    "parameters, has_family_history",
    [
        (
            {
                "p": 0.0,
            },
            False
        )
    ]
)
def test_family_history_has_family_history_of_asthma(parameters, has_family_history):
    family_history = FamilyHistory(probability=parameters["p"])
    assert family_history.has_family_history_of_asthma() == has_family_history
