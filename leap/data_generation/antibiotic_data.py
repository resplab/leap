import pathlib
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from leap.utils import get_data_path
from leap.data_generation.utils import get_province_id, get_sex_id
from leap.logger import get_logger
from typing import Tuple
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 2000
MAX_YEAR = 2019
MAX_AGE = 65



