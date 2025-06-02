import pandas as pd
import numpy as np
import itertools
from leap.utils import get_data_path
from leap.logger import get_logger
from leap.data_generation.occurrence_calibration_data import get_asthma_occurrence_prediction

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 1999
STABILIZATION_YEAR = 2025
MIN_ASTHMA_AGE = 3  # Minimum age for asthma diagnosis
MAX_ASTHMA_AGE = 62
MAX_AGE = 110



