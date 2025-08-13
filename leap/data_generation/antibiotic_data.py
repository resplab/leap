import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from leap.utils import get_data_path
from leap.data_generation.utils import get_province_id, get_sex_id, heaviside
from leap.logger import get_logger
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_YEAR = 2000
MAX_YEAR = 2019
MAX_AGE = 65


def estimate_alpha(
    df: pd.DataFrame,
    formula: str,
    offset: np.ndarray | None = None,
    maxiter: int = 5000
) -> float:
    r"""Estimate the alpha parameter for the negative binomial model.

    The :math:`\alpha` parameter is the dispersion parameter for the negative binomial model:

    .. math::

        \alpha := \dfrac{1}{\theta} = \dfrac{\sigma^2 - \mu}{\mu^2}
    
    Args:
        df: A Pandas dataframe with data to be fitted.
        formula: The formula for the GLM model. See the `statsmodels documentation
            <https://www.statsmodels.org/stable/examples/notebooks/generated/glm_formula.html>`_
            for more information.
        offset: The offset to use in the model, if desired.
        maxiter: The maximum number of iterations to perform while fitting the model.

    Returns:
        The estimated alpha parameter for the negative binomial model.
    """

    model = smf.negativebinomial(
        formula=formula, data=df, offset=offset
    )

    result = model.fit(maxiter=maxiter, method="nm")
    alpha = result.params.loc["alpha"]
    return alpha


def load_birth_data(
    province: str = "BC", min_year: int = 2000, max_year: int = 2018
) -> pd.DataFrame:
    """Load the StatCan birth data.
    
    Args:
        province: The province to load the data for.
        min_year: The minimum year to load the data for. Must be an integer in the range
            ``[1999, 2021]``.
        max_year: The maximum year to load the data for. Must be an integer in the range
            ``[1999, 2021]``, and ```max_year >= min_year``.

    Returns:
        A pandas dataframe with the number of births in a province, stratified by year and sex.
        Columns:
        
        * ``year (int)``: The calendar year.
        * ``province (str)``: The province name.
        * ``sex (str)``: One of ``M`` = male, ``F`` = female
        * ``n_birth (int)``: The number of births in the given year, province, and sex.

    """
    df = pd.read_csv(get_data_path("original_data/17100005.csv"))

    # rename columns
    df.rename(
        columns={"REF_DATE": "year", "GEO": "province", "SEX": "sex", "VALUE": "n_birth"},
        inplace=True
    )

    # select only the age = 0 age group and the years where min_year <= year <= max_year
    df = df.loc[
        (df["year"] >= min_year) & 
        (df["year"] <= max_year) & 
        (df["AGE_GROUP"] == "0 years")
    ]

    # select only the columns we need
    df = df[["year", "province", "sex", "n_birth"]]

    # convert province names to 2-letter province IDs and select the province
    df["province"] = df["province"].apply(get_province_id)
    df = df.loc[df["province"] == province]

    # convert sex to 1-letter ID ("F", "M", "B") and remove the "B" (both) rows
    df["sex"] = df["sex"].apply(get_sex_id)
    df = df.loc[df["sex"] != "B"]

    # convert N to integer
    df["n_birth"] = df["n_birth"].apply(lambda x: int(x))

    df.reset_index(drop=True, inplace=True)

    return df


def load_antibiotic_data() -> pd.DataFrame:
    """Load the antibiotic dose data.

    The antibiotic prescription data is from the BC Ministry of Health and contains the total
    number of courses of antibiotics dispensed to infants, stratified by year and sex, ranging from
    2000 to 2018.

    The birth data is from StatCan census data and contains the number of births in BC,
    stratified by year and sex.
    
    Returns:
        A Pandas dataframe. Columns:
        
        * ``year (int)``: The calendar year.
        * ``sex (str)``: One of ``M`` = male, ``F`` = female.
        * ``n_abx (int)``: The number total number of courses of antibiotics dispensed to
            infants in BC for the given year and sex.
        * ``n_birth (int)``: The number of births in BC for the given year and sex.

    """

    df_abx = pd.read_csv(get_data_path("original_data/private/bc_abx_dose_data.csv"))
    df_birth = load_birth_data()

    df_abx = pd.merge(
        df_abx,
        df_birth,
        how="left",
        on=["year", "sex"]
    )

    return df_abx

    

def generate_antibiotic_model(
    df_abx: pd.DataFrame,
    formula: str = "n_abx ~ year + sex + heaviside(year, 2005) * year",
    alpha: float = 1.0,
    maxiter: int = 1000
) -> GLMResultsWrapper:
    """Generate a generalized linear model for antibiotic dose.

    In this function, we fit a generalized linear model (GLM) to the antibiotic prescription data
    using the negative binomial family. The model predicts the number of courses of antibiotics
    dispensed to infants in BC, given the year and sex.

    For more details, see :ref:`antibiotic_exposure_model`.
    
    Args:
        df_abx: The antibiotic prescription data. Contains the following columns:

            * ``year (int)``: The calendar year.
            * ``sex (str)``: One of ``M`` = male, ``F`` = female.
            * ``n_abx (int)``: The number total number of courses of antibiotics dispensed to
              infants in BC for the given year and sex.
            * ``n_birth (int)``: The number of births in BC for the given year and sex.
            
        formula: The formula for the GLM model. See the `statsmodels documentation 
            <https://www.statsmodels.org/stable/examples/notebooks/generated/glm_formula.html>`_
            for more information.
        alpha: The alpha parameter for the negative binomial model. This is the dispersion
            parameter, which controls the variance of the model.
        maxiter: The maximum number of iterations to perform while fitting the model.
    
    Returns:
        The fitted ``GLM`` model.
    """

    df = df_abx.copy()

    # Convert sex string to 1 or 2
    df["sex"] = df.apply(
        lambda x: 1 if x["sex"] == "F" else 2,
        axis=1
    )

    # Fit the GLM model
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=np.log(df["n_birth"])
    )
    results = model.fit(maxiter=maxiter)

    print(results.summary())

    return results


def get_predicted_abx_data(
    model: GLMResultsWrapper,
    df: pd.DataFrame | None = None,
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR
) -> pd.DataFrame:
    """Get predicted data from a GLM model.

    The GLM model must be fitted on the following columns:

    * ``year (int)``: The calendar year.
    * ``sex (int)``: One of ``0`` = female, ``1`` = male.
    
    Args:
        model: The fitted GLM model for predicting the number of courses of antibiotics during
            the first year of life, given year and sex.
        df: (optional) If provided, the function will use this dataframe to predict the data. The
            dataframe must contain the following columns:

            * ``year (int)``: The calendar year.
            * ``sex (str)``: One of ``M`` = male, ``F`` = female.

            If not provided, the function will generate a dataframe with all combinations of
            year and sex in the range of ``min_year`` to ``max_year``.
        min_year: The minimum year to predict.
        max_year: The maximum year to predict.
        
    Returns:
        A dataframe containing the predicted number of antibiotics prescribed per person during
        infancy for a given birth year and sex.
        Columns:
        
        * ``year (int)``: The calendar year.
        * ``sex (str)``: One of ``M`` = male, ``F`` = female.
        * ``n_abx_μ (float)``: The predicted number of antibiotics prescribed per person during
          infancy for the given birth year and sex.
    """

    if df is None:
        df = pd.DataFrame(
            data=list(itertools.product(
                list(range(min_year, max_year + 1)), [1, 2]
            )),
            columns=["year", "sex"]
        )
    else:
        df["sex"] = df.apply(
            lambda x: 1 if x["sex"] == "F" else 2,
            axis=1
        )


    df["n_abx_μ"] = np.exp(model.predict(df, which="linear"))
    df["sex"] = df.apply(
        lambda x: "F" if x["sex"] == 1 else "M",
        axis=1
    )
    return df


def generate_antibiotic_data(
    return_type: str = "csv"
) -> GLMResultsWrapper | None:
    """Fit a ``GLM`` for antibiotic prescriptions in the first year of life and generate data.

    Args:
        return_type: The type of data to return. If ``csv``, the function will save a CSV file
            with the predicted data. If ``model``, the function will return the fitted GLM model.
    
    Returns:
        ``None`` if ``return_type`` is ``csv``, otherwise a fitted ``GLM`` model for predicting
        the number of antibiotic prescriptions during the first year of life.

    """
    formula = "n_abx ~ year + sex + heaviside(year, 2005) * year"
    df_abx = load_antibiotic_data()
    alpha = estimate_alpha(df_abx, formula, offset=np.log(df_abx["n_birth"]))
    model_abx = generate_antibiotic_model(df_abx, formula, alpha)
    if return_type == "csv":
        df_abx_pred = get_predicted_abx_data(model_abx)
        df_abx_pred.to_csv(
            get_data_path("processed_data/antibiotic_predictions.csv"),
            index=False
        )
    else:
        return model_abx


if __name__ == "__main__":
    generate_antibiotic_data()
