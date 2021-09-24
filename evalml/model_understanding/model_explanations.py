from evalml.model_understanding import calculate_permutation_importance
from evalml.utils import get_logger 

def explain(
    pipeline,
    X=None,
    y=None,
    importance_method="permutation",
    max_features=5,
    objective="auto",
):
    """ Outputs a human-readable explanation of trained pipeline behavior.

    Args:
        pipeline (PipelineBase): The pipeline to explain.
        X (pd.DataFrame): If importance_method is permutation, the holdout X data to compute importance with. Ignored otherwise.
        y (pd.Series): The holdout y data, used to obtain the name of the target class. If importance_method is permutation, used to compute importance with. 
        importance_method (str): The method of determining feature importance. One of ["permutation", "feature"]. Defaults to "permutation".
        max_features (int): The maximum number of features to include in an explanation. Defaults to 5.
        objective (str, ObjectiveBase): If importance_method is permutation, the objective to compute importance with. Ignored otherwise, defaults to "auto".

    """
    logger = get_logger(f"{__name__}.explain")

    if objective == "auto":    
        objective = {
            "binary": "Log Loss Binary",
            "multiclass": "Log Loss Multiclass",
            "regression": "R2",
            "time series regression": "R2",
            "time series binary": "Log Loss Binary",
            "time series multiclass": "Log Loss Multiclass",
        }[pipeline.problem_type.value]

    if importance_method == "permutation":
        imp_df = calculate_permutation_importance(pipeline, X, y, objective)
    elif importance_method == "feature":
        imp_df = pipeline.feature_importance()
    else:
        raise ValueError(f"Unknown importance method {importance_method}")

    most_important_features, somewhat_important_features, detrimental_features = get_influential_features(
        imp_df,
        max_features,
    )
    explanation = _fill_template(pipeline.estimator, y.name, objective, most_important_features, somewhat_important_features, detrimental_features)
    logger.info(explanation)


def get_influential_features(imp_df, max_features):
    
    # Separate negative and positive features
    neg_imp_df = imp_df[imp_df["importance"] < 0]
    pos_imp_df = imp_df[imp_df["importance"] >= 0]

    # Drop insignificant features
    pos_imp_df["importance"] = pos_imp_df["importance"]/sum(pos_imp_df["importance"])

    num_feats = min(len(pos_imp_df), max_features)
    imp_features = pos_imp_df[:num_feats]
    return (
      list(imp_features[imp_features["importance"]>=0.2]["feature"]),
      list(imp_features[imp_features["importance"].between(0.05, 0.2)]["feature"]),
      list(neg_imp_df["feature"])
      )


def _fill_template(estimator, target, objective, most_important, somewhat_important, detrimental_feats):
    if target is not None:
        beginning = f"{estimator}: The performance of predicting {target} as measured by {objective}"
    else:
        beginning = f"{estimator}: The {objective} performance"

    def enumerate_features(feature_list):
        text = "" if len(feature_list) == 2 else ","
        for i in range(1, len(feature_list)):
            if i == len(feature_list)-1:
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
    neither = ""
    if not (len(heavy) or len(somewhat)):
        neither = " is not strongly influenced by any single feature."
    else:
        neither = "."

    # Detrimental Description
    detrimental = ""
    if len(detrimental_feats) > 0:
        if len(detrimental_feats) == 1:
            detrimental = f" The feature {detrimental_feats[0]}"
            tag = "this feature."
        else:
            detrimental = f" The features {detrimental_feats[0]}"
            detrimental = detrimental + enumerate_features(detrimental_feats)
            tag = "these features."
        detrimental = detrimental + " detracted from model performance. We suggest removing " + tag

    return beginning + heavy + somewhat + neither + detrimental
