import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from leap.utils import get_data_path
from leap.logger import get_logger
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 2000
MAX_AGE = 65


def parse_age_group(x: str, max_age: int = MAX_AGE) -> Tuple[int, int]:
    """Parse an age group string into a tuple of integers.
    
    Args:
        x: The age group string. Must be in the format "X-Y", "X+", "X-Y years", "<1 year".

    Returns:
        A tuple of integers representing the lower and upper age of the age group.

    Examples:
    
        >>> parse_age_group("0-4")
        (0, 4)
        >>> parse_age_group("5-9 years")
        (5, 9)
        >>> parse_age_group("10+")
        (10, 100)
        >>> parse_age_group("<1 year")
        (0, 1)
    """
    if x == "<1 year":
        return 0, 1
    elif "-" in x:
        return int(x.split(" ")[0].split("-")[0]), int(x.split(" ")[0].split("-")[1])
    elif "+" in x:
        return int(x.split("+")[0]), max_age
    else:
        raise ValueError(f"Invalid age group: {x}")


def load_asthma_df(starting_year: int = STARTING_YEAR) -> pd.DataFrame:
    """Load the asthma incidence and prevalence data.

    Args:
        starting_year: The starting year for the data. Data before this year will be excluded
            from the analysis.
    
    Returns:
        The asthma incidence and prevalence data.
        Columns:
        
        * ``year (int)``: The calendar year.
        * ``age_group (str)``: The age group.
        * ``age (int)``: The average age of the age group.
        * ``sex (str)``: One of ``F`` = female, ``M`` = male.
        * ``incidence (float)``: The incidence of asthma.
        * ``prevalence (float)``: The prevalence of asthma.
    """

    df = pd.read_csv(get_data_path("original_data/private/asthma_inc_prev.csv"))

    # Rename columns
    df.rename(
        columns={
            "fiscal_year": "year", "age_group_desc": "age_group", "gender": "sex"
        },
        inplace=True
    )

    # Filter for year >= starting_year
    df = df.loc[df["year"] >= starting_year]

    # Age groups are in the format "X-Y" or "80+"
    # Set the age to the average of the age group
    df["age"] = df.apply(
        lambda x: int(np.mean(parse_age_group(x["age_group"]))), # type: ignore
        axis=1
    ) # type: ignore

    # Key assumption: asthma starts at age 3
    # Set incidence = prevalence at age 3
    df["incidence"] = df.apply(
        lambda x: x["prevalence"] if x["age"] == 3 else x["incidence"],
        axis=1
    )

    return df


def generate_occurrence_model(
    df_asthma: pd.DataFrame, formula: str, occ_type: str, maxiter: int = 1000
) -> GLMResultsWrapper:
    """Generate a ``GLM`` model for asthma incidence or prevalence.
    
    Args:
        df_asthma: The asthma dataframe. Must have columns:

            * ``year (int)``: The calendar year.
            * ``sex (str)``: One of ``M`` = male, ``F`` = female.
            * ``age (int)``: The age in years.
            * ``incidence (float)``: The incidence of asthma.
            * ``prevalence (float)``: The prevalence of asthma.
            
        formula: The formula for the GLM model. See the ```statsmodels`` documentation 
            <https://www.statsmodels.org/stable/examples/notebooks/generated/glm_formula.html>`_
            for more information.
        occ_type: The type of occurrence data to model. Must be one of ``"incidence"`` or
            ``"prevalence"``.
        maxiter: The maximum number of iterations to perform while fitting the model.
    
    Returns:
        The fitted ``GLM`` model.
    """

    # Create occurrence dataframe
    df = df_asthma[["year", "sex", "age", occ_type]].copy()

    # Convert sex string to 0 or 1
    df["sex"] = df.apply(
        lambda x: 0 if x["sex"] == "F" else 1,
        axis=1
    )

    # Fit the GLM model
    model = smf.glm(formula=formula, data=df, family=sm.families.Gaussian())
    results = model.fit(maxiter=maxiter)

    print(results.summary())

    return results


def generate_incidence_model(
    df_asthma: pd.DataFrame, maxiter: int = 1000
) -> GLMResultsWrapper:
    """Generate a ``GLM`` model for asthma incidence.
    
    Args:
    
        df_asthma: The asthma dataframe. Must have columns:

            * ``year (int)``: The calendar year.
            * ``sex (str)``: One of ``M`` = male, ``F`` = female.
            * ``age (int)``: The age in years.
            * ``incidence (float)``: The incidence of asthma.
            * ``prevalence (float)``: The prevalence of asthma.

        maxiter: The maximum number of iterations to perform while fitting the model.
    
    Returns:
        The fitted ``GLM`` model.
    """

    formula = "np.log(incidence) ~ sex*year + sex*poly(age, degree=5)"
    results = generate_occurrence_model(
        df_asthma, formula=formula, occ_type="incidence", maxiter=maxiter
    )
    return results


def generate_prevalence_model(
    df_asthma: pd.DataFrame, maxiter: int = 1000
) -> GLMResultsWrapper:
    """Generate a ``GLM`` model for asthma prevalence.
    
    Args:
    
        df_asthma: The asthma dataframe. Must have columns:

            * ``year (int)``: The calendar year.
            * ``sex (str)``: One of ``M`` = male, ``F`` = female.
            * ``age (int)``: The age in years.
            * ``incidence (float)``: The incidence of asthma.
            * ``prevalence (float)``: The prevalence of asthma.
        
        maxiter: The maximum number of iterations to perform while fitting the model.
    
    Returns:
        The fitted ``GLM`` model.
    """

    formula = "np.log(prevalence) ~ sex*poly(year, degree=2)*poly(age, degree=5)"
    results = generate_occurrence_model(
        df_asthma, formula=formula, occ_type="prevalence", maxiter=maxiter
    )
    return results


def generate_occurrence_data():
    df_asthma = load_asthma_df()
    incidence_model = generate_incidence_model(df_asthma)
    prevalence_model = generate_prevalence_model(df_asthma)


if __name__ == "__main__":
    generate_occurrence_data()