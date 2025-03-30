import pandas as pd
import numpy as np
import json
from leap.utils import get_data_path
from leap.logger import get_logger
from typing import Tuple
from leap.data_generation.rutils import ordinal_regression_r

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)


def compute_control_score(
    daytime_symptoms: int, nocturnal_symptoms: int, inhaler_use: int, limited_activities: int
) -> int:
    """Compute the control score based on the control data.

    Args:
        daytime_symptoms: Whether the person has daytime symptoms, 1 = True, 0 = False.
        nocturnal_symptoms: Whether the person has nocturnal symptoms, 1 = True, 0 = False.
        inhaler_use: Whether the person uses an inhaler for asthma, 1 = True, 0 = False.
        limited_activities: Whether the person's activites are limited due to asthma,
            1 = True, 0 = False.

    Returns:
        The control score, ranging from 0 to 4, with 0 being the best control and 4 being the worst.

    """
    control_score = (
        int(daytime_symptoms == 1) + 
        int(nocturnal_symptoms == 1) + 
        int(inhaler_use == 1) + 
        int(limited_activities == 1)
    )
    return control_score


def compute_control_level(control_score: int) -> int:
    """Compute the control level based on the control score.

    Use the GINA guidelines to determine the control level (Box 2-2, GINA guidelines 2020):
    `GINA guidelines 2020 <https://ginasthma.org/wp-content/uploads/2020/04/GINA-2020-full-report_-final-_wms.pdf>`_.

    Args:
        control_score: The control score, ranging from 0 to 4, with 0 being the best control and
            4 being the worst.

    Returns:
        The control level, ranging from 1 to 3.

        * 1 = fully-controlled
        * 2 = partially-controlled
        * 3 = uncontrolled

    """

    if control_score == 0:
        control_level = 1
    elif control_score < 3 and control_score > 0:
        control_level = 2
    elif control_score >= 3:
        control_level = 3
    else:
        raise ValueError(f"Invalid control score: {control_score}")
    return control_level



def load_control_data() -> pd.DataFrame:
    """Load the control data from the ``EBA`` study.

    Returns:
        A dataframe containing the control data from the ``EBA`` study.
        Columns:

        * ``patient_id (int)``: The 8-digit patient ID.
        * ``visit (int)``: The visit number. Visits were scheduled every 3 months for a year.
          A value in ``[1, 5]``.
        * ``age (float)``: The age of the person in years.
        * ``sex (int)``: 0 = female, 1 = male.
        * ``daytime_symptoms (int)``: Whether the person has daytime symptoms,
          ``1 = True``, ``0 = False``.
        * ``nocturnal_symptoms (int)``: Whether the person has nocturnal symptoms,
          ``1 = True``, ``0 = False``.
        * ``inhaler_use (int)``: Whether the person uses an inhaler for asthma,
          ``1 = True``, ``0 = False``.
        * ``limited_activities (int)``: Whether the person's activites are limited due to asthma,
          ``1 = True``, ``0 = False``.
        * ``age_at_asthma_dx (float)``: The age at which the person was diagnosed with asthma.
        * ``time_since_dx (float)``: The time since asthma diagnosis in years.
        * ``time_since_dx_cat``: TODO.
        * ``control_score (int)``: The control score, ranging from 0 to 4, with 0 being the best
          control and 4 being the worst.
        * ``control_score_missing (int)``: Whether any of the control scores are missing,
          ``1 = True``, ``0 = False``.
        * ``control_level (int)``: The control level, ranging from 1 to 3, where:

          * 1 = fully-controlled
          * 2 = partially-controlled
          * 3 = uncontrolled

        * ``exacerbations (int)``: TODO.
        * ``exac (int)``: Whether the person has had an exacerbation, ``1 = True``, ``0 = False``.


    """
    file_path = get_data_path("original_data/private/df_EBA.csv")
    df = pd.read_csv(file_path)

    # Convert the column names to snake_case
    df.rename(columns={
        "studyId": "patient_id",
        "daytimeSymptoms": "daytime_symptoms",
        "nocturnalSymptoms": "nocturnal_symptoms",
        "inhalerUse": "inhaler_use",
        "limitedActivities": "limited_activities",
        "ageAtAsthmaDx": "age_at_asthma_dx"
    }, inplace=True)
    df.columns = df.columns.str.lower()

    # Convert 1 = Yes, 2 = No to 1 = True, 0 = False
    df["daytime_symptoms"] = df["daytime_symptoms"].apply(
        lambda x: 1 if x == 1 else 0 if x == 2 else np.nan
    )
    df["nocturnal_symptoms"] = df["nocturnal_symptoms"].apply(
        lambda x: 1 if x == 1 else 0 if x == 2 else np.nan
    )
    df["inhaler_use"] = df["inhaler_use"].apply(
        lambda x: 1 if x == 1 else 0 if x == 2 else np.nan
    )
    df["limited_activities"] = df["limited_activities"].apply(
        lambda x: 1 if x == 1 else 0 if x == 2 else np.nan
    )

    # Compute the control score
    df["control_score"] = df.apply(
        lambda x: compute_control_score(
            x["daytime_symptoms"], x["nocturnal_symptoms"], x["inhaler_use"], x["limited_activities"]
        ),
        axis=1,
    )

    # Add a column to indicate if any of the control scores are missing
    df["control_score_missing"] = df.apply(
        lambda x: int(
            np.isnan(x["daytime_symptoms"]) or np.isnan(x["nocturnal_symptoms"]) or
            np.isnan(x["inhaler_use"]) or np.isnan(x["limited_activities"])
        ),
        axis=1
    )

    # Compute the control level
    df["control_level"] = df["control_score"].apply(compute_control_level)
    df["exac"] = df.apply(
        lambda x: int(x["exacerbations"] == 1),
        axis=1
    )

    # Reassign the patient_id
    patient_id_map = {
        patient_id: i for i, patient_id in enumerate(df["patient_id"].unique())
    }
    df["patient_id"] = df["patient_id"].apply(
        lambda x: patient_id_map[x]
    )
    return df


def fit_ordinal_regression_model(df: pd.DataFrame) -> Tuple[dict, dict, dict]:
    """Fit an ordinal regression model to the control data.

    Args:
        df: A dataframe containing information about asthma control levels from the EBA study:

            * ``patient_id (int)``: The 8-digit patient ID.
            * ``visit (int)``: The visit number. Visits were scheduled every 3 months for a year.
              A value in ``[1, 5]``.
            * ``age (float)``: The age of the person in years.
            * ``sex (int)``: 0 = female, 1 = male.
            * ``daytime_symptoms (int)``: Whether the person has daytime symptoms,
              ``1 = True``, ``0 = False``.
            * ``nocturnal_symptoms (int)``: Whether the person has nocturnal symptoms,
              ``1 = True``, ``0 = False``.
            * ``inhaler_use (int)``: Whether the person uses an inhaler for asthma,
              ``1 = True``, ``0 = False``.
            * ``limited_activities (int)``: Whether the person's activites are limited due to asthma,
              ``1 = True``, ``0 = False``.
            * ``age_at_asthma_dx (float)``: The age at which the person was diagnosed with asthma.
            * ``time_since_dx (float)``: The time since asthma diagnosis in years.
            * ``time_since_dx_cat``: TODO.
            * ``control_score (int)``: The control score, ranging from 0 to 4, with 0 being the best
              control and 4 being the worst.
            * ``control_score_missing (int)``: Whether any of the control scores are missing,
              ``1 = True``, ``0 = False``.
            * ``control_level (int)``: The control level, ranging from 1 to 3, where:

              * 1 = fully-controlled
              * 2 = partially-controlled
              * 3 = uncontrolled

            * ``exacerbations (int)``: TODO.
            * ``exac (int)``: Whether the person has had an exacerbation,
              ``1 = True``, ``0 = False``.

    Returns:
        A tuple containing three dictionaries.

        * The first dictionary contains the beta coefficients for the ordinal regression model.

        .. code-block::

            {
                "βsex": The coefficient for the sex term,
                "βage2": The coefficient for the age^2 term,
                "βsexage": The coefficient for the sex*age term,
                "βsexage2": The coefficient for the sex*age^2 term,
                "βage": The coefficient for the age term
            }

        * The second dictionary contains the thresholds for the ordinal regression model.

        .. code-block::

            {
                "θ_12": The threshold between control levels 1 and 2,
                "θ_23": The threshold between control levels 2 and 3
            }

        * The third dictionary contains the random effects for the ordinal regression model.

        .. code-block::

            {
                "β0_μ": The mean random effect for the intercept,
                "β0_σ": The standard deviation of the random effect
            }

    """
    df["age"] = df["age"] / 100
    df["age2"] = df["age"] ** 2
    formula = f"as.factor(control_level) ~ as.factor(sex)*age + as.factor(sex)*age2"
    coefficients, std = ordinal_regression_r(
        formula=formula,
        df=df,
        random="patient_id",
        hessian=True,
        link="logistic",
        nAGQ=5
    )
    
    beta_coefficients = {
        "βsex": coefficients["as.factor(sex)1"],
        "βage": coefficients["age"],
        "βage2": coefficients["age2"],
        "βsexage": coefficients["as.factor(sex)1:age"],
        "βsexage2": coefficients["as.factor(sex)1:age2"]
    }

    thresholds = {
        "θ_12": coefficients["1|2"],
        "θ_23": coefficients["2|3"]
    }

    random_effects = {
        "β0_μ": 0.0,
        "β0_σ": std
    }
    return beta_coefficients, thresholds, random_effects


def generate_control_data():
    """Generate the asthma control coefficients and thresholds."""

    # Load the control levels data from the EBA study
    df = load_control_data()

    # Fit an ordinal regression model to the control levels data
    beta_coefficients, thresholds, random_effects = fit_ordinal_regression_model(df)

    # Update the config file with the beta coefficients and thresholds
    config_path = get_data_path("processed_data/config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["control"]["parameters"] = {
        "θ": [thresholds["θ_12"], thresholds["θ_23"]],
        **beta_coefficients
    }
    config["control"]["hyperparameters"] = random_effects
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logger.message("Asthma control coefficients and thresholds generated and saved to config.json")


if __name__ == "__main__":
    generate_control_data()


