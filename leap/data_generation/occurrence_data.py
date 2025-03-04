import pathlib
import pandas as pd
import numpy as np
import plotly.express as px
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
MAX_YEAR = 2019
MAX_AGE = 65

def poly(
    x: list[float] | np.ndarray,
    degree: int = 1,
    alpha: list[float] | np.ndarray | None = None,
    norm2: list[float] | np.ndarray | None = None,
    orthogonal: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a polynomial basis for a vector.

    See `Orthogonal polynomial regression in Python
    <https://davmre.github.io/blog/python/2013/12/15/orthogonal_poly/>`_ for more
    information on this function.
    
    Args:
        x: The vector to generate the polynomial basis for.
        degree: The degree of the polynomial.
        orthogonal: Whether to generate an orthogonal polynomial basis.
        
    Returns:
        The polynomial basis, as a 2D Numpy array. If ``orthogonal`` is ``True``, the function
        will return a tuple of three Numpy arrays: the orthogonal polynomial basis, the alpha
        values, and the norm2 values.

    Examples:

        >>> poly([1, 2, 3], degree=2) # doctest: +NORMALIZE_WHITESPACE
        array([[1, 1],
               [2, 4],
               [3, 9]])

        >>> poly([1, 2, 3], degree=2, orthogonal=True) # doctest: +NORMALIZE_WHITESPACE
        (array([[-7.07106781e-01,  4.08248290e-01],
               [-5.55111512e-17, -8.16496581e-01],
               [ 7.07106781e-01,  4.08248290e-01]]), array([2., 2.]), array([3.        , 2.        , 0.66666667]))
    """
    n = degree + 1
    x = np.asarray(x).flatten()
    if alpha is not None and norm2 is not None:
        Z = np.empty((len(x), n))
        Z[:,0] = 1
        if degree > 0:
            Z[:, 1] = x - alpha[0]
        if degree > 1:
            for i in np.arange(1, degree):
                Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
        Z /= np.sqrt(norm2)
        return Z[:, 1:]
    else:
        if(degree >= len(np.unique(x))):
            raise ValueError("'degree' must be less than number of unique points")
        if orthogonal:
            xbar = np.mean(x)
            x = x - xbar
            X = np.vander(x, n, increasing=True)
            Q, R = np.linalg.qr(X)

            Z = np.diag(np.diag(R))
            raw = np.dot(Q, Z)

            norm2 = np.sum(raw**2, axis=0)
            alpha = (
                np.sum((raw**2) * np.reshape(x, (-1, 1)), axis=0)/norm2 + 
                xbar
            )[:degree]
            Z = raw / np.sqrt(norm2)
            return Z[:, 1:], alpha, norm2
        else:
            X = np.vander(x, n, increasing=True)
            return X[:, 1:]


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
        (10, 65)
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

    _, alpha, norm2 = poly(df_asthma["age"].to_list(), degree=5, orthogonal=True)

    formula = "np.log(incidence) ~ sex*year + sex*poly(age, degree=5, alpha=alpha, norm2=norm2)"
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

    _, alpha_age, norm2_age = poly(df_asthma["age"].to_list(), degree=5, orthogonal=True)
    _, alpha_year, norm2_year = poly(df_asthma["year"].to_list(), degree=2, orthogonal=True)
    formula = "np.log(prevalence) ~ sex*poly(year, degree=2, alpha=alpha_year, norm2=norm2_year)" + \
        "*poly(age, degree=5, alpha=alpha_age, norm2=norm2_age)"
    results = generate_occurrence_model(
        df_asthma, formula=formula, occ_type="prevalence", maxiter=maxiter
    )
    return results


def plot_occurrence(
    df: pd.DataFrame,
    y: str,
    title: str = "",
    file_path: pathlib.Path | None = None,
    min_year: int = STARTING_YEAR,
    max_year: int = MAX_YEAR,
    year_interval: int = 2,
    max_age: int = 110,
    width: int = 1000,
    height: int = 800
):
    """Plot the incidence or prevalence of asthma.
    
    Args:
        df: A dataframe containing either incidence or prevalence data. Must have columns:

            * ``year (int)``: The calendar year.
            * ``sex (str)``: One of ``F`` = female, ``M`` = male.
            * ``age (int)``: The age in years.
            * y: Specified by the ``y`` argument, this will be the y data.
            * y_pred: Optional, the predicted y data. If this column is present, it will be plotted
              alongside the actual data. The column name must be the same as ``y`` with ``_pred``
              appended. For example, if ``y`` is ``incidence``, then the predicted data must be
              ``incidence_pred``.

        y: The name of the column in the dataframe which will be plotted as the y data.
        title: The title of the plot.
        file_path: The path to save the plot to. If ``None``, the plot will be displayed.
        min_year: The minimum year to plot.
        max_year: The maximum year to plot.
        year_interval: The interval between years. This is used if you don't want to plot every year.
        max_age: The maximum age to plot.
        width: The width of the plot.
        height: The height of the plot.
        
    Returns:
        If ``file_path`` is ``None``, the plot will be displayed. Otherwise, the plot will be saved
        to the specified path.
    """

    years = np.arange(min_year, max_year + 1, step=year_interval)
    fig = px.line(
        df.loc[(df["year"].isin(years)) & (df["age"] <= max_age)].dropna(),
        x="age",
        y=y,
        color="year",
        markers=True,
        facet_col="sex",
        title=title
    )
    if f"{y}_pred" in df.columns:
        fig.add_traces(
            px.line(
                df.loc[(df["year"].isin(years)) & (df["age"] <= max_age)],
                x="age",
                y=f"{y}_pred",
                color="year",
                markers=True,
                facet_col="sex",
                title=title
            ).data
        )
    if file_path is None:
        fig.show()
    else:
        fig.write_image(str(file_path), width=width, height=height, scale=2)


def generate_occurrence_data():
    df_asthma = load_asthma_df()
    incidence_model = generate_incidence_model(df_asthma)
    prevalence_model = generate_prevalence_model(df_asthma)


if __name__ == "__main__":
    generate_occurrence_data()