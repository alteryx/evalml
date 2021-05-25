import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import Bunch

from evalml.objectives.utils import get_objective
from evalml.problem_types import is_classification
from evalml.utils import infer_feature_types


def calculate_permutation_importance(pipeline, X, y, objective, n_repeats=5, n_jobs=None, random_seed=0):
    """Calculates permutation importance for features.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (pd.DataFrame): The input data used to score and compute permutation importance
        y (pd.Series): The target data
        objective (str, ObjectiveBase): Objective to score on
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    Returns:
        pd.DataFrame, Mean feature importance scores over a number of shuffles.
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    objective = get_objective(objective, return_instance=True)
    if not objective.is_defined_for_problem_type(pipeline.problem_type):
        raise ValueError(f"Given objective '{objective.name}' cannot be used with '{pipeline.name}'")

    if pipeline._supports_fast_permutation_importance:
        perm_importance = _fast_permutation_importance(pipeline, X, y, objective,
                                                       n_repeats=n_repeats,
                                                       n_jobs=n_jobs,
                                                       random_seed=random_seed)
    else:
        perm_importance = _slow_permutation_importance(pipeline, X, y, objective,
                                                       n_repeats=n_repeats,
                                                       n_jobs=n_jobs,
                                                       random_seed=random_seed)

    mean_perm_importance = perm_importance["importances_mean"]
    feature_names = list(X.columns)
    mean_perm_importance = list(zip(feature_names, mean_perm_importance))
    mean_perm_importance.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(mean_perm_importance, columns=["feature", "importance"])


def calculate_permutation_importance_one_column(X, y, pipeline, col_name, objective, n_repeats=5, fast=True, precomputed_features=None, random_seed=0):
    """Calculates permutation importance for one column in the original dataframe.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (pd.DataFrame): The input data used to score and compute permutation importance
        y (pd.Series): The target data
        objective (str, ObjectiveBase): Objective to score on
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        fast (bool): Whether to use the fast method of calculating the permutation importance or not.
        precomputed_features (pd.DataFrame): Precomputed features necessary to calculate permutation importance using the fast method. Defaults to None.

        random_seed (int): Seed for the random number generator. Defaults to 0.
    Returns:
        pd.DataFrame, Mean feature importance scores over a number of shuffles.
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)
    objective = get_objective(objective, return_instance=True)

    if fast:
        if precomputed_features is None:
            raise ValueError("Fast method of calculating permutation importance requires precomputed_features")
        if is_classification(pipeline.problem_type):
            y = pipeline._encode_targets(y)

        def scorer(pipeline, features, y, objective):
            if objective.score_needs_proba:
                preds = pipeline.estimator.predict_proba(features)
            else:
                preds = pipeline.estimator.predict(features)
            score = pipeline._score(X, y, preds, objective)
            return score if objective.greater_is_better else -score

        baseline_score = scorer(pipeline, precomputed_features, y, objective)
        scores = _calculate_permutation_scores_fast(
            pipeline, precomputed_features, y, objective, col_name, random_seed, n_repeats, scorer, baseline_score,
        )
        importances = baseline_score - np.array(scores)
        return np.mean(importances)
    else:
        def scorer(pipeline, X, y):
            scores = pipeline.score(X, y, objectives=[objective])
            return scores[objective.name] if objective.greater_is_better else -scores[objective.name]

        baseline_score = scorer(pipeline, X, y)
        scores = _calculate_permutation_scores_slow(pipeline, X, y, col_name, random_seed, n_repeats, scorer)
        importances = baseline_score - np.array(scores)
        return np.mean(importances)


def _fast_permutation_importance(pipeline, X, y, objective, n_repeats=5, n_jobs=None, random_seed=None):
    """Calculate permutation importance faster by only computing the estimator features once.

    Only used for pipelines that support this optimization.
    """

    precomputed_features = pipeline.compute_estimator_features(X, y)

    if is_classification(pipeline.problem_type):
        y = pipeline._encode_targets(y)

    def scorer(pipeline, features, y, objective):
        if objective.score_needs_proba:
            preds = pipeline.estimator.predict_proba(features)
        else:
            preds = pipeline.estimator.predict(features)
        score = pipeline._score(X, y, preds, objective)
        return score if objective.greater_is_better else -score

    baseline_score = scorer(pipeline, precomputed_features, y, objective)

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores_fast)(
        pipeline, precomputed_features, y, objective, col_name, random_seed, n_repeats, scorer, baseline_score,
    ) for col_name in X.columns)

    importances = baseline_score - np.array(scores)
    return {'importances_mean': np.mean(importances, axis=1)}


def _calculate_permutation_scores_fast(pipeline, precomputed_features, y, objective, col_name,
                                       random_seed, n_repeats, scorer, baseline_score):
    """Calculate the permutation score when `col_name` is permuted."""
    random_state = np.random.RandomState(random_seed)

    scores = np.zeros(n_repeats)

    # If column is not in the features or provenance, assume the column was dropped
    if col_name not in precomputed_features.columns and col_name not in pipeline._get_feature_provenance():
        return scores + baseline_score

    if col_name in precomputed_features.columns:
        col_idx = precomputed_features.columns.get_loc(col_name)
    else:
        col_idx = [precomputed_features.columns.get_loc(col) for col in pipeline._get_feature_provenance()[col_name]]

    return _shuffle_and_score_helper(precomputed_features, n_repeats, random_state, col_idx, scorer, True, pipeline, y, objective)


def _slow_permutation_importance(pipeline, X, y, objective, n_repeats=5, n_jobs=None, random_seed=None):
    def scorer(pipeline, X, y):
        scores = pipeline.score(X, y, objectives=[objective])
        return scores[objective.name] if objective.greater_is_better else -scores[objective.name]

    baseline_score = scorer(pipeline, X, y)

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores_slow)(
        pipeline, X, y, col_idx, random_seed, n_repeats, scorer
    ) for col_idx in range(X.shape[1]))

    importances = baseline_score - np.array(scores)
    perm_importance = Bunch(importances_mean=np.mean(importances, axis=1),
                            importances_std=np.std(importances, axis=1),
                            importances=importances)

    return perm_importance


def _calculate_permutation_scores_slow(estimator, X, y, col_name,
                                       random_seed, n_repeats, scorer):
    """Calculate score when `col_idx` is permuted."""
    random_state = np.random.RandomState(random_seed)
    col_idx = col_name
    if col_name in X.columns:
        col_idx = X.columns.get_loc(col_name)
    return _shuffle_and_score_helper(X, n_repeats, random_state, col_idx, scorer, False, estimator, y, None)


def _shuffle_and_score_helper(X_features, n_repeats, random_state, col_idx, scorer, is_fast, pipeline, y, objective):
    scores = np.zeros(n_repeats)

    # This is what sk_permutation_importance does. Useful for thread safety
    X_permuted = X_features.copy()
    shuffling_idx = np.arange(X_features.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        col = X_permuted.iloc[shuffling_idx, col_idx]
        col.index = X_permuted.index
        X_permuted.iloc[:, col_idx] = col
        if is_fast:
            feature_score = scorer(pipeline, X_permuted, y, objective)
        else:
            feature_score = scorer(pipeline, X_permuted, y)
        scores[n_round] = feature_score
    return scores


def _slow_scorer(pipeline, X, y, objective):
    scores = pipeline.score(X, y, objectives=[objective])
    return scores[objective.name] if objective.greater_is_better else -scores[objective.name]


def _fast_scorer(pipeline, features, X, y, objective):
    if objective.score_needs_proba:
        preds = pipeline.estimator.predict_proba(features)
    else:
        preds = pipeline.estimator.predict(features)
    score = pipeline._score(X, y, preds, objective)
    return score if objective.greater_is_better else -score
