"""Permutation importance methods."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evalml.objectives.utils import get_objective
from evalml.problem_types import is_classification
from evalml.problem_types.utils import is_regression
from evalml.utils import import_or_raise, infer_feature_types, jupyter_check


def calculate_permutation_importance(
    pipeline,
    X,
    y,
    objective,
    n_repeats=5,
    n_jobs=None,
    random_seed=0,
):
    """Calculates permutation importance for features.

    Args:
        pipeline (PipelineBase or subclass): Fitted pipeline.
        X (pd.DataFrame): The input data used to score and compute permutation importance.
        y (pd.Series): The target data.
        objective (str, ObjectiveBase): Objective to score on.
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        pd.DataFrame: Mean feature importance scores over a number of shuffles.

    Raises:
        ValueError: If objective cannot be used with the given pipeline.
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    objective = get_objective(objective, return_instance=True)
    if not objective.is_defined_for_problem_type(pipeline.problem_type):
        raise ValueError(
            f"Given objective '{objective.name}' cannot be used with '{pipeline.name}'",
        )

    if pipeline._supports_fast_permutation_importance:
        precomputed_features = pipeline.transform_all_but_final(X, y)
        perm_importance = _fast_permutation_importance(
            pipeline,
            X,
            y,
            objective,
            precomputed_features,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )
    else:
        perm_importance = _slow_permutation_importance(
            pipeline,
            X,
            y,
            objective,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )

    mean_perm_importance = perm_importance["importances_mean"]
    feature_names = list(X.columns)
    mean_perm_importance = list(zip(feature_names, mean_perm_importance))
    mean_perm_importance.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(mean_perm_importance, columns=["feature", "importance"])


def graph_permutation_importance(pipeline, X, y, objective, importance_threshold=0):
    """Generate a bar graph of the pipeline's permutation importance.

    Args:
        pipeline (PipelineBase or subclass): Fitted pipeline.
        X (pd.DataFrame): The input data used to score and compute permutation importance.
        y (pd.Series): The target data.
        objective (str, ObjectiveBase): Objective to score on.
        importance_threshold (float, optional): If provided, graph features with a permutation importance whose absolute value is larger than importance_threshold. Defaults to 0.

    Returns:
        plotly.Figure, a bar graph showing features and their respective permutation importance.

    Raises:
        ValueError: If importance_threshold is not greater than or equal to 0.
    """
    go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    perm_importance = calculate_permutation_importance(pipeline, X, y, objective)
    perm_importance["importance"] = perm_importance["importance"]

    if importance_threshold < 0:
        raise ValueError(
            f"Provided importance threshold of {importance_threshold} must be greater than or equal to 0",
        )
    # Remove features with close to zero importance
    perm_importance = perm_importance[
        abs(perm_importance["importance"]) >= importance_threshold
    ]
    # List is reversed to go from ascending order to descending order
    perm_importance = perm_importance.iloc[::-1]

    title = "Permutation Importance"
    subtitle = (
        "The relative importance of each input feature's "
        "overall influence on the pipelines' predictions, computed using "
        "the permutation importance algorithm."
    )
    data = [
        go.Bar(
            x=perm_importance["importance"],
            y=perm_importance["feature"],
            orientation="h",
        ),
    ]

    layout = {
        "title": "{0}<br><sub>{1}</sub>".format(title, subtitle),
        "height": 800,
        "xaxis_title": "Permutation Importance",
        "yaxis_title": "Feature",
        "yaxis": {"type": "category"},
    }

    fig = go.Figure(data=data, layout=layout)
    return fig


def calculate_permutation_importance_one_column(
    pipeline,
    X,
    y,
    col_name,
    objective,
    n_repeats=5,
    fast=True,
    precomputed_features=None,
    random_seed=0,
):
    """Calculates permutation importance for one column in the original dataframe.

    Args:
        pipeline (PipelineBase or subclass): Fitted pipeline.
        X (pd.DataFrame): The input data used to score and compute permutation importance.
        y (pd.Series): The target data.
        col_name (str, int): The column in X to calculate permutation importance for.
        objective (str, ObjectiveBase): Objective to score on.
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        fast (bool): Whether to use the fast method of calculating the permutation importance or not. Defaults to True.
        precomputed_features (pd.DataFrame): Precomputed features necessary to calculate permutation importance using the fast method. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        float: Mean feature importance scores over a number of shuffles.

    Raises:
        ValueError: If pipeline does not support fast permutation importance calculation.
        ValueError: If precomputed_features is None.
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)
    objective = get_objective(objective, return_instance=True)

    if fast:
        if not pipeline._supports_fast_permutation_importance:
            raise ValueError(
                "Pipeline does not support fast permutation importance calculation",
            )
        if precomputed_features is None:
            raise ValueError(
                "Fast method of calculating permutation importance requires precomputed_features",
            )
        permutation_importance = _fast_permutation_importance(
            pipeline,
            X,
            y,
            objective,
            precomputed_features,
            col_name=col_name,
            n_repeats=n_repeats,
            random_seed=random_seed,
        )
    else:
        permutation_importance = _slow_permutation_importance(
            pipeline,
            X,
            y,
            objective,
            col_name=col_name,
            n_repeats=n_repeats,
            random_seed=random_seed,
        )
    return permutation_importance["importances_mean"]


def _fast_permutation_importance(
    pipeline,
    X,
    y,
    objective,
    precomputed_features,
    col_name=None,
    n_repeats=5,
    n_jobs=None,
    random_seed=None,
):
    """Calculate permutation importance faster by only computing the estimator features once.

    Only used for pipelines that support this optimization.
    """
    if is_classification(pipeline.problem_type):
        y = pipeline._encode_targets(y)
    baseline_score = _fast_scorer(pipeline, precomputed_features, X, y, objective)
    if col_name is None:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_permutation_scores_fast)(
                pipeline,
                precomputed_features,
                y,
                objective,
                col_name,
                random_seed,
                n_repeats,
                _fast_scorer,
                baseline_score,
            )
            for col_name in X.columns
        )
        importances = baseline_score - np.array(scores)
        return {"importances_mean": np.mean(importances, axis=1)}
    else:
        scores = _calculate_permutation_scores_fast(
            pipeline,
            precomputed_features,
            y,
            objective,
            col_name,
            random_seed,
            n_repeats,
            _fast_scorer,
            baseline_score,
        )
    importances = baseline_score - np.array(scores)
    importances_mean = (
        np.mean(importances, axis=1) if col_name is None else np.mean(importances)
    )
    return {"importances_mean": importances_mean}


def _calculate_permutation_scores_fast(
    pipeline,
    precomputed_features,
    y,
    objective,
    col_name,
    random_seed,
    n_repeats,
    scorer,
    baseline_score,
):
    """Calculate the permutation score when `col_name` is permuted."""
    random_state = np.random.RandomState(random_seed)
    scores = np.zeros(n_repeats)

    # If column is not in the features or provenance, assume the column was dropped
    if (
        col_name not in precomputed_features.columns
        and col_name not in pipeline._get_feature_provenance()
    ):
        return scores + baseline_score

    if col_name in precomputed_features.columns:
        col_idx = precomputed_features.columns.get_loc(col_name)
    else:
        col_idx = [
            precomputed_features.columns.get_loc(col)
            for col in pipeline._get_feature_provenance()[col_name]
        ]

    return _shuffle_and_score_helper(
        pipeline,
        precomputed_features,
        y,
        objective,
        col_idx,
        n_repeats,
        scorer,
        random_state,
        is_fast=True,
    )


def _slow_permutation_importance(
    pipeline,
    X,
    y,
    objective,
    col_name=None,
    n_repeats=5,
    n_jobs=None,
    random_seed=None,
):
    """If `col_name` is not None, calculates permutation importance for only the column with that name.

    Otherwise, calculates the permutation importance for all columns in the input dataframe.
    """
    baseline_score = _slow_scorer(pipeline, X, y, objective)
    if col_name is None:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_permutation_scores_slow)(
                pipeline,
                X,
                y,
                col_idx,
                objective,
                _slow_scorer,
                n_repeats,
                random_seed,
            )
            for col_idx in range(X.shape[1])
        )
    else:
        baseline_score = _slow_scorer(pipeline, X, y, objective)
        scores = _calculate_permutation_scores_slow(
            pipeline,
            X,
            y,
            col_name,
            objective,
            _slow_scorer,
            n_repeats,
            random_seed,
        )
    importances = baseline_score - np.array(scores)
    importances_mean = (
        np.mean(importances, axis=1) if col_name is None else np.mean(importances)
    )
    return {"importances_mean": importances_mean}


def _calculate_permutation_scores_slow(
    estimator,
    X,
    y,
    col_name,
    objective,
    scorer,
    n_repeats,
    random_seed,
):
    """Calculate score when `col_idx` is permuted."""
    random_state = np.random.RandomState(random_seed)
    col_idx = col_name
    if col_name in X.columns:
        col_idx = X.columns.get_loc(col_name)
    return _shuffle_and_score_helper(
        estimator,
        X,
        y,
        objective,
        col_idx,
        n_repeats,
        scorer,
        random_state,
        is_fast=False,
    )


def _shuffle_and_score_helper(
    pipeline,
    X_features,
    y,
    objective,
    col_idx,
    n_repeats,
    scorer,
    random_state,
    is_fast=True,
):
    scores = np.zeros(n_repeats)

    # This is what sk_permutation_importance does. Useful for thread safety
    X_permuted = X_features.copy()
    shuffling_idx = np.arange(X_features.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        col = X_permuted.iloc[shuffling_idx, col_idx]
        col.index = X_permuted.index
        X_permuted.iloc[:, col_idx] = col
        X_permuted.ww.init(schema=X_features.ww.schema)
        if is_fast:
            feature_score = scorer(pipeline, X_permuted, X_features, y, objective)
        else:
            feature_score = scorer(pipeline, X_permuted, y, objective)
        scores[n_round] = feature_score
    return scores


def _slow_scorer(pipeline, X, y, objective):
    scores = pipeline.score(X, y, objectives=[objective])
    return (
        scores[objective.name]
        if objective.greater_is_better
        else -scores[objective.name]
    )


def _fast_scorer(pipeline, features, X, y, objective):
    if objective.score_needs_proba:
        preds = pipeline.estimator.predict_proba(features)
    else:
        preds = pipeline.estimator.predict(features)
        if is_regression(pipeline.problem_type):
            preds = pipeline.inverse_transform(preds)
    score = pipeline._score(X, y, preds, objective)
    return score if objective.greater_is_better else -score
