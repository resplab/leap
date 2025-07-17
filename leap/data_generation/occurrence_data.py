import pathlib
import pandas as pd
import numpy as np
import itertools
import json
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from leap.utils import get_data_path, poly
from leap.data_generation.utils import parse_age_group
from leap.logger import get_logger
from typing import Tuple, Dict, Any
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_YEAR = 2000
MAX_YEAR = 2019
MAX_AGE = 65


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
        lambda x: int(np.mean(parse_age_group(x["age_group"], max_age=MAX_AGE))), # type: ignore
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
            
        formula: The formula for the GLM model. See the `statsmodels documentation 
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

    # Convert sex string to 1 or 2
    df["sex"] = df.apply(
        lambda x: 1 if x["sex"] == "F" else 2,
        axis=1
    )

    # Fit the GLM model
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    results = model.fit(maxiter=maxiter)

    print(results.summary())

    # Create prediction dataframe
    df_pred = pd.DataFrame(
        data=list(itertools.product(list(range(2000, 2066)), [1, 2], list(range(3, 65)))),
        columns=["year", "sex", "age"]
    )

    occurrence_pred = results.predict(df_pred, which="mean", transform=True)
    df_pred[f"{occ_type}_pred"] = occurrence_pred

    df = pd.merge(df, df_pred, on=["year", "sex", "age"], how="outer")
    df["sex"] = df.apply(
        lambda x: "F" if x["sex"] == 1 else "M",
        axis=1
    )

    plot_occurrence(
        df, y=occ_type, title=f"Asthma {occ_type.capitalize()} per 100 in BC",
        file_path=get_data_path(f"data_generation/figures/asthma_{occ_type}_comparison.png")
    )

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

    formula = "incidence ~ sex*year + " + \
        f"sex*poly(age, degree=5, alpha={list(alpha)}, norm2={list(norm2)})"
    results = generate_occurrence_model(
        df_asthma, formula=formula, occ_type="incidence", maxiter=maxiter
    )
    return results


def generate_prevalence_model(
    df_asthma: pd.DataFrame, maxiter: int = 1000
) -> Tuple[GLMResultsWrapper, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        A tuple containing:
        1. The fitted ``GLM`` model.
        2. The alpha parameters for the age polynomial.
        3. The norm2 parameters for the age polynomial.
        4. The alpha parameters for the year polynomial.
        5. The norm2 parameters for the year polynomial.
    """

    _, alpha_age, norm2_age = poly(df_asthma["age"].to_list(), degree=5, orthogonal=True)
    _, alpha_year, norm2_year = poly(df_asthma["year"].to_list(), degree=2, orthogonal=True)
    formula = "prevalence ~ " + \
        f"sex*poly(year, degree=2, alpha={list(alpha_year)}, norm2={list(norm2_year)})" + \
        f"*poly(age, degree=5, alpha={list(alpha_age)}, norm2={list(norm2_age)})"
    results = generate_occurrence_model(
        df_asthma, formula=formula, occ_type="prevalence", maxiter=maxiter
    )
    return results, alpha_age, norm2_age, alpha_year, norm2_year

def get_predicted_data(
    model: GLMResultsWrapper,
    pred_col: str,
    min_age: int = 3,
    max_age: int = 100,
    min_year: int = STARTING_YEAR,
    max_year: int = MAX_YEAR
) -> pd.DataFrame:
    """Get predicted data from a GLM model.

    The GLM model must be fitted on the following columns:

    * ``year (int)``: The calendar year.
    * ``sex (int)``: One of ``0`` = female, ``1`` = male.
    * ``age (int)``: The age in years.
    
    Args:
        model: The fitted GLM model.
        pred_col: The name of the column to store the predicted data.
        min_age: The minimum age to predict.
        max_age: The maximum age to predict.
        min_year: The minimum year to predict.
        max_year: The maximum year to predict.
        
    Returns:
        A dataframe containing the predicted data.
        Columns:
        
        * ``year (int)``: The calendar year.
        * ``sex (str)``: One of ``M`` = male, ``F`` = female.
        * ``age (int)``: The age in years.
        * ``pred_col (float)``: The predicted data.
    """

    df = pd.DataFrame(
        data=list(itertools.product(
            list(range(min_year, max_year)), [1, 2], list(range(min_age, max_age))
        )),
        columns=["year", "sex", "age"]
    )

    df[pred_col] = np.exp(model.predict(df, which="linear"))
    df["sex"] = df.apply(
        lambda x: "F" if x["sex"] == 1 else "M",
        axis=1
    )
    return df


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
            * ``y (float)``: Specified by the ``y`` argument, this will be the y data.
            * ``y_pred (float)``: Optional, the predicted y data. If this column is present, it will
              be plotted alongside the actual data. The column name must be the same as ``y`` with
              ``_pred`` appended. For example, if ``y`` is ``incidence``, then the predicted data
              must be ``incidence_pred``.

        y: The name of the column in the dataframe which will be plotted as the ``y`` data.
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

def add_beta_parameters(
    model: GLMResultsWrapper, parameter_map: Dict[str, list[int]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Add the beta parameters to the config dictionary.
    
    Args:
        model: The fitted GLM model.
        parameter_map: A dictionary mapping the parameter names to their indices in the model
            parameters field, ``model.params``. The keys are the parameter names and the
            values are lists of indices. For example, if ``βyear`` is the second parameter in the
            list, then the mapping would be ``{"βyear": [1]}``, and it would be accessed by
            ``model.params.iloc[1]``.
        config: The config dictionary to add the parameters to.
        
    Returns:
        The config dictionary with the beta parameters added.
    """
    beta_parameters = {}
    for name, indices in parameter_map.items():
        beta_parameters[name] = model.params.iloc[indices].to_list()
        if len(beta_parameters[name]) == 1:
            beta_parameters[name] = beta_parameters[name][0]

    # add the beta parameters to the config dictionary
    for key, value in beta_parameters.items():
        config[key] = value
    return config


def generate_occurrence_data():
    """Generate the asthma incidence and prevalence data.
    
    Saves the data to a CSV file: ``processed_data/asthma_occurrence_predictions.csv``.

    The data is also plotted and saved to the following files:
    
    * ``data_generation/figures/asthma_incidence_predicted.png``: The predicted asthma
      incidence per 100 in BC.
    * ``data_generation/figures/asthma_prevalence_predicted.png``: The predicted asthma
      prevalence per 100 in BC.

    """
    df_asthma = load_asthma_df()
    incidence_model = generate_incidence_model(df_asthma)
    prevalence_model, alpha_age, norm2_age, alpha_year, norm2_year = generate_prevalence_model(df_asthma)
    df_incidence = get_predicted_data(
        incidence_model, "incidence", max_age=110, min_year=1999, max_year=2066
    )
    df_prevalence = get_predicted_data(
        prevalence_model, "prevalence", max_age=110, min_year=1999, max_year=2066
    )
    df = pd.merge(df_incidence, df_prevalence, on=["year", "sex", "age"], how="left")
    plot_occurrence(
        df,
        y="incidence",
        title="Predicted Asthma Incidence per 100 in BC",
        min_year=2000,
        max_year=2025,
        year_interval=5,
        max_age=63,
        file_path=get_data_path("data_generation/figures/asthma_incidence_predicted.png")
    )
    plot_occurrence(
        df,
        y="prevalence",
        title="Predicted Asthma Prevalence per 100 in BC",
        min_year=2000,
        max_year=2025,
        year_interval=5,
        max_age=63,
        file_path=get_data_path("data_generation/figures/asthma_prevalence_predicted.png")
    )
    df.to_csv(get_data_path("processed_data/asthma_occurrence_predictions.csv"), index=False)


    with open(get_data_path("processed_data/config.json"), "r") as f:
        config = json.load(f)

    config["incidence"]["parameters"] = add_beta_parameters(
        incidence_model,
        {
            "β0": [0],
            "βsex": [1],
            "βyear": [2],
            "βsexyear": [3],
            "βage": list(range(4, 9)),
            "βsexage": list(range(9, 14))
        },
        config["incidence"]["parameters"]
    )
    config["prevalence"]["parameters"] = add_beta_parameters(
        prevalence_model, {
            "β0": [0],
            "βsex": [1],
            "βyear": [2, 3],
            "βsexyear": [4, 5],
            "βage": list(range(6, 11)),
            "βsexage": list(range(11, 16)),
            "βyearage": list(range(16, 26)),
            "βsexyearage": list(range(26, 36))
        },
        config["prevalence"]["parameters"]
    )

    config["prevalence"]["poly_parameters"] = {
        "alpha_age": list(alpha_age),
        "norm2_age": list(norm2_age),
        "alpha_year": list(alpha_year),
        "norm2_year": list(norm2_year)
    }
    config["incidence"]["poly_parameters"] = {
        "alpha_age": list(alpha_age),
        "norm2_age": list(norm2_age)
    }

    with open(get_data_path("processed_data/config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    generate_occurrence_data()