"""Human Readable Pipeline Explanations."""
import pandas as pd

import evalml
from evalml.model_family import ModelFamily
from evalml.model_understanding import calculate_permutation_importance
from evalml.utils import get_logger, infer_feature_types


def readable_explanation(
    pipeline,
    X=None,
    y=None,
    importance_method="permutation",
    max_features=5,
    min_importance_threshold=0.05,
    objective="auto",
):
    """Outputs a human-readable explanation of trained pipeline behavior.

    Args:
        pipeline (PipelineBase): The pipeline to explain.
        X (pd.DataFrame): If importance_method is permutation, the holdout X data to compute importance with. Ignored otherwise.
        y (pd.Series): The holdout y data, used to obtain the name of the target class. If importance_method is permutation, used to compute importance with.
        importance_method (str): The method of determining feature importance. One of ["permutation", "feature"]. Defaults to "permutation".
        max_features (int): The maximum number of influential features to include in an explanation. This does not affect the number of detrimental features reported. Defaults to 5.
        min_importance_threshold (float): The minimum percent of total importance a single feature can have to be considered important. Defaults to 0.05.
        objective (str, ObjectiveBase): If importance_method is permutation, the objective to compute importance with. Ignored otherwise, defaults to "auto".

    Raises:
        ValueError: if any arguments passed in are invalid or the pipeline is not fitted.
    """
    logger = get_logger(f"{__name__}.explain")

    if not pipeline._is_fitted:
        raise ValueError(
            "Pipelines must be fitted in order to run feature explanations.",
        )

    if min_importance_threshold >= 1 or min_importance_threshold < 0:
        raise ValueError(
            f"The minimum importance threshold must be a percentage value in the range [0, 1), not {min_importance_threshold}.",
        )

    if importance_method == "permutation":

        if objective == "auto":
            objective = evalml.automl.get_default_primary_search_objective(
                pipeline.problem_type,
            )

        if X is None or y is None:
            raise ValueError(
                "X and y are required parameters for explaining pipelines with permutation importance.",
            )

        X = infer_feature_types(X)
        y = infer_feature_types(y)
        imp_df = calculate_permutation_importance(pipeline, X, y, objective)
    elif importance_method == "feature":
        objective = None
        imp_df = pipeline.feature_importance
    else:
        raise ValueError(f"Unknown importance method {importance_method}.")

    linear_importance = False
    if (
        pipeline.estimator.model_family == ModelFamily.LINEAR_MODEL
        and importance_method == "feature"
    ):
        linear_importance = True

    (
        most_important_features,
        somewhat_important_features,
        detrimental_features,
    ) = get_influential_features(
        imp_df,
        max_features,
        min_importance_threshold,
        linear_importance,
    )
    target = y if y is None else y.name
    explanation = _fill_template(
        pipeline.estimator,
        target,
        objective,
        most_important_features,
        somewhat_important_features,
        detrimental_features,
    )
    logger.info(explanation)


def get_influential_features(
    imp_df,
    max_features=5,
    min_importance_threshold=0.05,
    linear_importance=False,
):
    """Finds the most influential features as well as any detrimental features from a dataframe of feature importances.

    Args:
        imp_df (pd.DataFrame): DataFrame containing feature names and associated importances.
        max_features (int): The maximum number of features to include in an explanation. Defaults to 5.
        min_importance_threshold (float): The minimum percent of total importance a single feature can have to be considered important. Defaults to 0.05.
        linear_importance (bool): When True, negative feature importances are not considered detrimental. Defaults to False.

    Returns:
        (list, list, list): Lists of feature names corresponding to heavily influential, somewhat influential, and detrimental features, respectively.

    """
    heavy_importance_threshold = max(0.2, min_importance_threshold + 0.1)

    # Separate negative and positive features, if situation calls
    if linear_importance:
        pos_imp_df = imp_df
        pos_imp_df["importance"] = abs(pos_imp_df["importance"])
        neg_imp_df = pd.DataFrame({"feature": [], "importance": []})
    else:
        neg_imp_df = imp_df[imp_df["importance"] < 0]
        pos_imp_df = imp_df[imp_df["importance"] >= 0]

    # Normalize the positive features to sum to 1
    pos_imp_df["importance"] = pos_imp_df["importance"] / sum(pos_imp_df["importance"])

    num_feats = min(len(pos_imp_df), max_features)
    imp_features = pos_imp_df[:num_feats]

    heavy_importance = imp_features[
        imp_features["importance"] >= heavy_importance_threshold
    ]
    somewhat_importance = imp_features[
        imp_features["importance"] < heavy_importance_threshold
    ]
    return (
        list(heavy_importance["feature"]),
        list(
            somewhat_importance[
                somewhat_importance["importance"] >= min_importance_threshold
            ]["feature"],
        ),
        list(neg_imp_df["feature"]),
    )


def _fill_template(
    estimator,
    target,
    objective,
    most_important,
    somewhat_important,
    detrimental_feats,
):
    # Get the objective to a printable string
    if objective is not None:
        if isinstance(objective, evalml.objectives.ObjectiveBase):
            objective = objective.name
        if objective != "R2":  # Remove any title case if necessary
            objective = objective.lower()

    # Beginning of description
    objective_str = f" as measured by {objective}" if objective is not None else ""
    beginning = (
        f"{estimator}: The output{objective_str}"
        if target is None
        else f"{estimator}: The prediction of {target}{objective_str}"
    )

    def enumerate_features(feature_list):
        text = "" if len(feature_list) == 2 else ","
        for i in range(1, len(feature_list)):
            if i == len(feature_list) - 1:
                text = text + f" and {feature_list[i]}"
            else:
                text = text + f" {feature_list[i]},"
        return text

    # Heavily influential description
    heavy = ""
    if len(most_important) > 0:
        heavy = f" is heavily influenced by {most_important[0]}"
        if len(most_important) > 1:
            heavy = heavy + enumerate_features(most_important)
        if len(somewhat_important) > 0:
            heavy = heavy + ", and"

    # Somewhat influential description
    somewhat = ""
    if len(somewhat_important) > 0:
        somewhat = f" is somewhat influenced by {somewhat_important[0]}"
        if len(somewhat_important) > 1:
            somewhat = somewhat + enumerate_features(somewhat_important)

    # Neither!
    neither = "."
    if not (len(heavy) or len(somewhat)):
        neither = " is not strongly influenced by any single feature. Lower the `min_importance_threshold` to see more."

    # Detrimental Description
    detrimental = ""
    if len(detrimental_feats) > 0:
        if len(detrimental_feats) == 1:
            detrimental = f"\nThe feature {detrimental_feats[0]}"
            tag = "this feature."
        else:
            detrimental = f"\nThe features {detrimental_feats[0]}"
            detrimental = detrimental + enumerate_features(detrimental_feats)
            tag = "these features."
        detrimental = (
            detrimental
            + " detracted from model performance. We suggest removing "
            + tag
        )

    return beginning + heavy + somewhat + neither + detrimental
