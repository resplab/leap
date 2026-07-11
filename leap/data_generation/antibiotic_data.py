import pandas as pd
import numpy as np
import datetime as dt
import json
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from leap.utils import get_data_path, TimeDelta
from leap.data_generation.utils import get_province_id, get_sex_id, heaviside, get_parser
from leap.logger import get_logger
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MIN_TIMEPOINT = dt.datetime(2000, 1, 1)
MAX_TIMEPOINT = dt.datetime(2019, 12, 31)
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
    time_delta_od: TimeDelta = TimeDelta(years=1),
    province: str = "BC",
    min_timepoint: dt.datetime = dt.datetime(2000, 1, 1),
    max_timepoint: dt.datetime = dt.datetime(2018, 12, 31)
) -> pd.DataFrame:
    """Load the StatCan birth data.
    
    Args:
        time_delta_od: The duration of the time intervals in the original antibiotic data.
        province: The province to load the data for.
        min_timepoint: The minimum timepoint to load the data for. Must be a ``datetime`` object in
            the range ``[1999, 2021]``.
        max_timepoint: The maximum timepoint to load the data for. Must be a ``datetime`` object in
            the range ``[1999, 2021]``, and ``max_timepoint >= min_timepoint``.

    Returns:
        A pandas dataframe with the number of births in a province, stratified by timepoint and sex.
        Columns:
        
        * ``timepoint (dt.datetime)``: The date and time.
        * ``province (str)``: The province name.
        * ``sex (str)``: One of ``M`` = male, ``F`` = female
        * ``n_birth (int)``: The number of births in the given year, province, and sex.

    """

    time_delta_tag = get_time_delta_tag(time_delta_od)
    df = pd.read_csv(
        get_data_path(f"processed_data/{time_delta_tag}/birth/birth_estimate.csv"),
        parse_dates=["timepoint"]
    )

    # select only the province and the timepoints where min_timepoint <= timepoint <= max_timepoint
    df = df.loc[
        (df["timepoint"] >= min_timepoint) & 
        (df["timepoint"] <= max_timepoint) & 
        (df["province"] == province)
    ]

    # pivot table to have separate rows for M and F
    df["F"] = df["N"] * (1 - df["prop_male"])
    df["M"] = df["N"] * df["prop_male"]
    df = df.melt(
        id_vars=["timepoint", "province"],
        value_vars=["F", "M"],
        var_name="sex",
        value_name="n_birth"
    )

    # convert N to integer
    df["n_birth"] = df["n_birth"].astype("int")

    df.reset_index(drop=True, inplace=True)

    return df


def load_antibiotic_data(time_delta: TimeDelta) -> pd.DataFrame:
    """Load the antibiotic dose data.

    The antibiotic prescription data is from the BC Ministry of Health and contains the total
    number of courses of antibiotics dispensed to infants, stratified by year and sex, ranging from
    2000 to 2018.

    The birth data is from StatCan census data and contains the number of births in BC,
    stratified by timepoint and sex.

    Args:
        time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years, etc.
    
    Returns:
        A Pandas dataframe. Columns:
        
        * ``timepoint (dt.datetime)``: The date and time.
        * ``sex (str)``: One of ``M`` = male, ``F`` = female.
        * ``n_abx (int)``: The number total number of courses of antibiotics dispensed to
            infants in BC for the given year and sex.
        * ``n_birth (int)``: The number of births in BC for the given year and sex.

    """

    df_abx = pd.read_csv(
        get_data_path("original_data/private/bc_abx_dose_data.csv"),
        parse_dates=["year"]
    )
    df_abx.rename(columns={"year": "timepoint"}, inplace=True)

    df_birth = load_birth_data()

    df_abx = pd.merge(
        df_abx,
        df_birth,
        how="left",
        on=["timepoint", "sex"]
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
    min_year: dt.datetime = MIN_TIMEPOINT,
    max_year: dt.datetime = MAX_TIMEPOINT
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
    time_delta: TimeDelta,
    return_type: str = "csv"
) -> GLMResultsWrapper | None:
    """Fit a ``GLM`` for antibiotic prescriptions in the first year of life and generate data.

    Args:
        time_delta: The duration of the time intervals to use for the data, e.g. 1 year, 5 years, etc.
        return_type: The type of data to return. If ``csv``, the function will save a CSV file
            with the predicted data. If ``model``, the function will return the fitted GLM model.
    
    Returns:
        ``None`` if ``return_type`` is ``csv``, otherwise a fitted ``GLM`` model for predicting
        the number of antibiotic prescriptions during the first year of life.

    """
    formula = "n_abx ~ year + sex + heaviside(year, 2005) * year"
    df_abx = load_antibiotic_data(time_delta)
    alpha = estimate_alpha(df_abx, formula, offset=np.log(df_abx["n_birth"]))
    model_abx = generate_antibiotic_model(df_abx, formula, alpha)
    if return_type == "csv":
        time_delta_tag = get_time_delta_tag(time_delta)
        df_abx_pred.to_csv(
            get_data_path(f"processed_data/{time_delta_tag}/antibiotic_predictions.csv"),
            index=False
        )

        # Update the config file with the beta coefficients and thresholds
        config_path = get_data_path(
            f"processed_data/{time_delta_tag}/config.json",
            mkdirs=True
        )
        with open(config_path) as f:
            config = json.load(f)

        config["antibiotic_exposure"]["parameters"]["β0"] = model_abx.params["Intercept"]
        config["antibiotic_exposure"]["parameters"]["βyear"] = model_abx.params["year"]
        config["antibiotic_exposure"]["parameters"]["βsex"] = model_abx.params["sex"]
        config["antibiotic_exposure"]["parameters"]["β2005"] = model_abx.params["heaviside(year, 2005)"]
        config["antibiotic_exposure"]["parameters"]["β2005_year"] = model_abx.params["heaviside(year, 2005):year"]
        config["antibiotic_exposure"]["parameters"]["θ"] = 1 / alpha

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        logger.message("Antibiotic exposure coefficients generated and saved to config.json")
    else:
        return model_abx


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    time_delta = TimeDelta(iso_string=args.time_delta)
    generate_antibiotic_data(time_delta)
