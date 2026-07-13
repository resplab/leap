import pathlib
import pandas as pd
import numpy as np
import datetime as dt
import itertools
import json
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from leap.utils import get_data_path, poly, date_range, get_time_delta_tag, TimeDelta
from leap.data_generation.utils import parse_age_group, get_parser, convert_numeric_to_sex, \
    convert_sex_to_numeric
from leap.logger import get_logger
from typing import Tuple, Dict, Any
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

STARTING_TIMEPOINT = dt.datetime(2000, 1, 1)
MAX_TIMEPOINT = dt.datetime(2019, 12, 31)
MAX_AGE = 65


def load_asthma_df(starting_timepoint: dt.datetime = STARTING_TIMEPOINT) -> pd.DataFrame:
    """Load the asthma incidence and prevalence data.

    Args:
        starting_timepoint: The starting date / time for the data. Data before this timepoint will
            be excluded from the analysis.
    
    Returns:
        The asthma incidence and prevalence data.
        Columns:
        
        * ``timepoint (dt.datetime)``: The date and time.
        * ``age_group (str)``: The age group.
        * ``age (int)``: The average age of the age group.
        * ``sex (str)``: One of ``F`` = female, ``M`` = male.
        * ``incidence (float)``: The incidence of asthma.
        * ``prevalence (float)``: The prevalence of asthma.
    """

    df = pd.read_csv(
        get_data_path("original_data/private/asthma_inc_prev.csv"),
        parse_dates=["fiscal_year"]
    )

    # Rename columns
    df.rename(
        columns={
            "fiscal_year": "timepoint", "age_group_desc": "age_group", "gender": "sex"
        },
        inplace=True
    )

    # Filter for timepoint >= starting_timepoint
    df = df.loc[df["timepoint"] >= starting_timepoint]

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

            * ``timepoint (dt.datetime)``: The date and time.
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
    df = df_asthma[["timepoint", "sex", "age", occ_type]].copy()

    # Convert sex string to 1 or 2
    df["sex"] = df["sex"].apply(convert_sex_to_numeric)

    # Fit the GLM model
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    results = model.fit(maxiter=maxiter)

    print(results.summary())

    # Create prediction dataframe
    df_pred = pd.DataFrame(
        data=list(itertools.product(list(range(2000, 2066)), [1, 2], list(range(3, 65)))),
        columns=["timepoint", "sex", "age"]
    )

    occurrence_pred = results.predict(df_pred, which="mean", transform=True)
    df_pred[f"{occ_type}_pred"] = occurrence_pred

    df = pd.merge(df, df_pred, on=["timepoint", "sex", "age"], how="outer")
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

            * ``timepoint (dt.datetime)``: The date and time.
            * ``sex (str)``: One of ``M`` = male, ``F`` = female.
            * ``age (int)``: The age in years.
            * ``incidence (float)``: The incidence of asthma.
            * ``prevalence (float)``: The prevalence of asthma.

        maxiter: The maximum number of iterations to perform while fitting the model.
    
    Returns:
        The fitted ``GLM`` model.
    """

    _, alpha, norm2 = poly(df_asthma["age"].to_list(), degree=5, orthogonal=True)

    formula = "incidence ~ sex*timepoint + " + \
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

            * ``timepoint (dt.datetime)``: The date and time.
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
        4. The alpha parameters for the timepoint polynomial.
        5. The norm2 parameters for the timepoint polynomial.
    """

    _, alpha_age, norm2_age = poly(df_asthma["age"].to_list(), degree=5, orthogonal=True)
    _, alpha_time, norm2_time = poly(df_asthma["timepoint"].to_list(), degree=2, orthogonal=True)
    formula = "prevalence ~ " + \
        f"sex*poly(timepoint, degree=2, alpha={list(alpha_time)}, norm2={list(norm2_time)})" + \
        f"*poly(age, degree=5, alpha={list(alpha_age)}, norm2={list(norm2_age)})"
    results = generate_occurrence_model(
        df_asthma, formula=formula, occ_type="prevalence", maxiter=maxiter
    )
    return results, alpha_age, norm2_age, alpha_time, norm2_time

def get_predicted_data(
    model: GLMResultsWrapper,
    pred_col: str,
    time_delta: TimeDelta,
    min_age: int = 3,
    max_age: int = 100,
    min_timepoint: dt.datetime = STARTING_TIMEPOINT,
    max_timepoint: dt.datetime = MAX_TIMEPOINT
) -> pd.DataFrame:
    """Get predicted data from a GLM model.

    The GLM model must be fitted on the following columns:

    * ``timepoint (dt.datetime)``: The date and time.
    * ``sex (int)``: One of ``0`` = female, ``1`` = male.
    * ``age (int)``: The age in years.
    
    Args:
        model: The fitted GLM model.
        pred_col: The name of the column to store the predicted data.
        time_delta: The duration of time between subsequent timepoints in the data, e.g. 1 month,
            1 year, etc.
        min_age: The minimum age to predict.
        max_age: The maximum age to predict.
        min_timepoint: The minimum timepoint to predict.
        max_timepoint: The maximum timepoint to predict.
        
    Returns:
        A dataframe containing the predicted data.
        Columns:
        
        * ``timepoint (dt.datetime)``: The date and time.
        * ``sex (str)``: One of ``M`` = male, ``F`` = female.
        * ``age (int)``: The age in years.
        * ``pred_col (float)``: The predicted data.
    """

    df = pd.DataFrame(
        data=list(itertools.product(
            list(date_range(min_timepoint, max_timepoint, time_delta)),
            [1, 2],
            list(range(min_age, max_age))
        )),
        columns=["timepoint", "sex", "age"]
    )

    df[pred_col] = np.exp(model.predict(df, which="linear"))
    df["sex"] = df["sex"].apply(convert_numeric_to_sex)
    return df


def plot_occurrence(
    df: pd.DataFrame,
    y: str,
    title: str = "",
    file_path: pathlib.Path | None = None,
    min_timepoint: dt.datetime = STARTING_TIMEPOINT,
    max_timepoint: dt.datetime = MAX_TIMEPOINT,
    time_interval: TimeDelta = TimeDelta(years=2),
    max_age: int = 110,
    width: int = 1000,
    height: int = 800
):
    """Plot the incidence or prevalence of asthma.
    
    Args:
        df: A dataframe containing either incidence or prevalence data. Must have columns:

            * ``timepoint (dt.datetime)``: The date and time.
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
        min_timepoint: The minimum timepoint to plot.
        max_timepoint: The maximum timepoint to plot.
        time_interval: The interval between years. This is used if you don't want to plot every year.
        max_age: The maximum age to plot.
        width: The width of the plot.
        height: The height of the plot.
        
    Returns:
        If ``file_path`` is ``None``, the plot will be displayed. Otherwise, the plot will be saved
        to the specified path.
    """

    timepoints = list(date_range(min_timepoint, max_timepoint + time_interval, step=time_interval))
    fig = px.line(
        df.loc[(df["timepoint"].isin(timepoints)) & (df["age"] <= max_age)].dropna(),
        x="age",
        y=y,
        color="timepoint",
        markers=True,
        facet_col="sex",
        title=title
    )
    if f"{y}_pred" in df.columns:
        fig.add_traces(
            px.line(
                df.loc[(df["timepoint"].isin(timepoints)) & (df["age"] <= max_age)],
                x="age",
                y=f"{y}_pred",
                color="timepoint",
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
            values are lists of indices. For example, if ``βtime`` is the second parameter in the
            list, then the mapping would be ``{"βtime": [1]}``, and it would be accessed by
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


def generate_occurrence_data(time_delta: TimeDelta):
    """Generate the asthma incidence and prevalence data.
    
    Saves the data to a CSV file: ``processed_data/asthma_occurrence_predictions.csv``.

    The data is also plotted and saved to the following files:
    
    * ``data_generation/figures/asthma_incidence_predicted.png``: The predicted asthma
      incidence per 100 in BC.
    * ``data_generation/figures/asthma_prevalence_predicted.png``: The predicted asthma
      prevalence per 100 in BC.

    Args:
        time_delta: The duration of time between subsequent timepoints in the data, e.g. 1 month,
            1 year, etc.

    """
    time_delta_tag = get_time_delta_tag(time_delta)
    df_asthma = load_asthma_df()
    incidence_model = generate_incidence_model(df_asthma)
    prevalence_model, alpha_age, norm2_age, alpha_time, norm2_time = generate_prevalence_model(df_asthma)
    df_incidence = get_predicted_data(
        incidence_model,
        "incidence",
        time_delta,
        max_age=110,
        min_timepoint=dt.datetime(1999, 1, 1),
        max_timepoint=dt.datetime(2066, 12, 31)
    )
    df_prevalence = get_predicted_data(
        prevalence_model,
        "prevalence",
        time_delta,
        max_age=110,
        min_timepoint=dt.datetime(1999, 1, 1),
        max_timepoint=dt.datetime(2066, 12, 31)
    )
    df = pd.merge(df_incidence, df_prevalence, on=["timepoint", "sex", "age"], how="left")
    plot_occurrence(
        df,
        y="incidence",
        title="Predicted Asthma Incidence per 100 in BC",
        min_timepoint=dt.datetime(2000, 1, 1),
        max_timepoint=dt.datetime(2025, 12, 31),
        time_interval=TimeDelta(years=5),
        max_age=63,
        file_path=get_data_path(
            f"data_generation/figures/{time_delta_tag}/asthma_incidence_predicted.png",
            mkdirs=True
        )
    )
    plot_occurrence(
        df,
        y="prevalence",
        title="Predicted Asthma Prevalence per 100 in BC",
        min_timepoint=dt.datetime(2000, 1, 1),
        max_timepoint=dt.datetime(2025, 12, 31),
        time_interval=TimeDelta(years=5),
        max_age=63,
        file_path=get_data_path(
            f"data_generation/figures/{time_delta_tag}/asthma_prevalence_predicted.png",
            mkdirs=True
        )
    )
    df.to_csv(
        get_data_path(
            f"processed_data/{time_delta_tag}/asthma_occurrence_predictions.csv", mkdirs=True
        ),
        index=False
    )


    with open(get_data_path(f"processed_data/{time_delta_tag}/config.json"), "r") as f:
        config = json.load(f)

    config["incidence"]["parameters"] = add_beta_parameters(
        incidence_model,
        {
            "β0": [0],
            "βsex": [1],
            "βtime": [2],
            "βsextime": [3],
            "βage": list(range(4, 9)),
            "βsexage": list(range(9, 14))
        },
        config["incidence"]["parameters"]
    )
    config["prevalence"]["parameters"] = add_beta_parameters(
        prevalence_model, {
            "β0": [0],
            "βsex": [1],
            "βtime": [2, 3],
            "βsextime": [4, 5],
            "βage": list(range(6, 11)),
            "βsexage": list(range(11, 16)),
            "βtimeage": list(range(16, 26)),
            "βsextimeage": list(range(26, 36))
        },
        config["prevalence"]["parameters"]
    )

    config["prevalence"]["poly_parameters"] = {
        "alpha_age": list(alpha_age),
        "norm2_age": list(norm2_age),
        "alpha_time": list(alpha_time),
        "norm2_time": list(norm2_time)
    }
    config["incidence"]["poly_parameters"] = {
        "alpha_age": list(alpha_age),
        "norm2_age": list(norm2_age)
    }

    with open(get_data_path(f"processed_data/{time_delta_tag}/config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    time_delta = TimeDelta(iso_string=args.time_delta)
    generate_occurrence_data(time_delta)