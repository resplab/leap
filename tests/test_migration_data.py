import pytest
import datetime as dt
import numpy as np
import pandas as pd
import itertools
from leap.data_generation.migration_data import get_delta_n,  \
    MIN_TIMEPOINT
from leap.logger import get_logger
from leap.utils import TimeDelta, date_range, PROJECTION_SCENARIOS, PROVINCE_MAP

logger = get_logger(__name__)



@pytest.fixture
def df_projection():
    df_projection = pd.DataFrame(
        list(itertools.product(
            list(PROVINCE_MAP.values())[0:2],
            PROJECTION_SCENARIOS,
            list(date_range(dt.datetime(2025, 1, 1), dt.datetime(2027, 1, 1), TimeDelta(months=1))),
            np.arange(0, 4, 1/12),
            ["F", "M"]
        )),
        columns=[
            "province", "projection_scenario", "timepoint", "age", "sex"
        ]
    )
    df_projection["N"] = np.random.randint(1000, 10000, df_projection.shape[0])
    df_projection["prob_death"] = np.random.uniform(0.0001, 0.01, df_projection.shape[0])
    return df_projection


@pytest.mark.parametrize(
    "n, n_prev, prob_death",
    [
        (1000, 1100, 0.01),
        (1000, 900, 0.01)
    ]
)
def test_get_delta_n(n, n_prev, prob_death):
    delta_n = get_delta_n(n=n, n_prev=n_prev, prob_death=prob_death)
    assert np.abs(delta_n) <= n



