import pandas as pd
from typing import Tuple
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import Formula

# import R's "base" package
base = importr("base")

# import R's "utils" package
utils = importr("utils")

# R package names
PACKAGE_NAMES = ["ordinal"]


def install_packages(package_names: list[str] = PACKAGE_NAMES) -> None:
    """Install R packages.
    
    This function checks if the specified R packages are installed, and if not,
    installs them from ``CRAN``. It uses the ``utils`` package to select a ``CRAN`` mirror
    and install the packages.
    
    Args:
        package_names: A list of R package names to install. Default is ["ordinal"].
    """
    package_names = [x for x in package_names if not rpackages.isinstalled(x)]
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    for package_name in package_names:
        utils.install_packages(package_name)


def ordinal_regression_r(
    formula: str,
    df: pd.DataFrame,
    random: str,
    hessian: bool = True,
    link: str = "logistic",
    nagq: int = 5
) -> Tuple[pd.Series, float]:
    """Fit an ordinal regression model using R.
    
    This function uses ``rpy2`` to interface with R and fit an ordinal regression model
    using the ``ordinal`` package. The model is fitted using the ``clmm2`` function, which
    fits a cumulative link mixed model, allowing for random effects.

    See:
    `ordinal::clmm2
    <https://www.rdocumentation.org/packages/ordinal/versions/2023.12-4.1/topics/clmm2>`_
    and
    `ordinal::clm2
    <https://www.rdocumentation.org/packages/ordinal/versions/2023.12-4.1/topics/clm2>`_
    for more details on the model and its parameters.

    Args:
        formula: The formula for the model, e.g. "response ~ predictor1 + predictor2".
        df: The data frame containing the data.
        random: The name of the random effect variable. This must be a column in the dataframe.
        hessian: Whether to compute the Hessian matrix (the inverse of the observed information
            matrix). Use ``hessian=True`` if you intend to call ``summary`` or ``vcov`` on the fit
            and ``hessian=False` in all other instances to save computing time. The argument is
            ignored if ``method="Newton"``, where the Hessian is always computed and returned.
            Default is ``True``. 
        link: The link function to use. Default is ``"logistic"``. Must be one of:

            * ``"logistic"``
            * ``"probit"``
            * ``"cloglog"``
            * ``"loglog"``
            * ``"cauchit"``
            * ``"Aranda-Ordaz"``
            * ``"log-gamma"``

        nagq: The number of adaptive Gauss-Hermite quadrature points. Default is ``5``.

    Returns:
        A tuple containing two values.

        The coefficients: A pandas Series containing the estimated coefficients.
        The standard deviation of the random effect: A float value.
    """

    ordinal = importr("ordinal")

    df[random] = df[random].apply(
        lambda x: str(x)
    ).astype("category")

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)

    model = ordinal.clmm2(
        location=Formula(formula),
        data=r_df,
        Hess=hessian,
        link=link,
        nAGQ=nagq,
        random=base.as_name(random)
    )

    coefficients = model.rx2("coefficients")
    std = model.rx2("stDev")
    coefficient_names = base.as_data_frame_list(base.names(coefficients))
    with localconverter(robjects.default_converter + pandas2ri.converter):
        model = robjects.conversion.rpy2py(model)
        coefficients = robjects.conversion.rpy2py(coefficients)
        coefficient_names = robjects.conversion.rpy2py(coefficient_names)
        std = robjects.conversion.rpy2py(std)[0]

    coefficient_names = coefficient_names.iloc[0].to_list()
    coefficient_names = ["intercept" if x == "" else x for x in coefficient_names]
    coefficients = pd.Series(coefficients, index=coefficient_names)
    return coefficients, std
    


# Install packages on import

install_packages()