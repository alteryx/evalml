from pdb import set_trace

import numpy as np
import pandas as pd
import woodwork as ww

from evalml.problem_types import is_binary, is_regression


def _get_cloned_feature_pipelines(
    features,
    X,
    pipeline,
    variable_has_features_passed_to_estimator,
):
    # --> long term: put flag on the object

    for component in pipeline.component_graph.component_instances.values():
        X, pipeline_params = component._handle_partial_dependence_fast_mode(
            X,
            pipeline.parameters,
        )
        pass

    # --> add docstring - and maybe typehinting?
    # mock out y for pipeline fitting
    len_X = len(X)
    if is_regression(pipeline.problem_type):
        # --> might want to use random seed here, though the mocked y shouldn't have an impact on PD
        mock_y = pd.Series(np.random.randint(0, 10, len_X))
    elif is_binary(pipeline.problem_type):
        mock_y = pd.Series(np.random.choice([True, False], size=len_X))
    else:
        mock_y = pd.Series(np.random.choice([0, 1, 2], size=len_X))

    # If DFSTransformer is present, confirm features parameter is specified and all features are present
    # in X
    dfs_transformer = pipeline.parameters.get("DFS Transformer")
    if dfs_transformer is not None:
        dfs_features = dfs_transformer["features"]
        X_cols = set(X.columns)
        if dfs_features is None or any(
            f.get_name() not in X_cols for f in dfs_features
        ):
            # set_trace()
            raise ValueError(
                "Cannot use fast mode with DFS Transformer when features are unspecified or not all present in X.",
            )

    # Handle components that use feature importance to remove some features before passing into estimator
    new_parameters = pipeline.parameters
    selector = None
    if "RF Regressor Select From Model" in pipeline.parameters:
        # --> can I be more generic and just look for any selector that does this kind of thing? Are there others than these two?
        selector = "RF Regressor Select From Model"
    elif "RF Classifier Select From Model" in pipeline.parameters:
        selector = "RF Classifier Select From Model"
    if selector is not None:
        # Feature selector's shouldn't drop any columns for the cloned pipeline
        new_parameters[selector]["percent_features"] = 1.0
        new_parameters[selector]["threshold"] = 0.0
    # --> check if dfs transformer is present, and if all the features aren't in X, raise an error

    # Create a fit pipeline for each feature
    cloned_feature_pipelines = {}
    for variable in features:
        # Don't fit pipelines if the feature has no impact on predictions
        if not variable_has_features_passed_to_estimator[variable]:
            continue
        pipeline_copy = pipeline.new(
            parameters=new_parameters,
        )
        pipeline_copy.fit(X.ww[[variable]], mock_y)
        cloned_feature_pipelines[variable] = pipeline_copy
    # --> maybe check if not passed to estimator earlier and return then
    return cloned_feature_pipelines


def _transform_single_feature(
    X,
    X_t,
    feature_provenance,
    variable,
    part_dep_column,
    cloned_pipeline,
):
    changed_col_df = pd.DataFrame({variable: part_dep_column})
    changed_col_df.ww.init(
        logical_types={variable: X.ww.logical_types[variable]},
    )

    # Take the changed column and send it through transform by itself
    X_t_single_col = cloned_pipeline.transform_all_but_final(changed_col_df)
    cols_to_replace = [variable]
    if feature_provenance.get(variable):
        # cols to replace has to be in the same order as X_t
        cols_to_replace = [
            col for col in X_t if col in feature_provenance.get(variable)
        ]
    # --> not keeping in woodwork - problematic?

    # If some categories get dropped, they won't be in X_t, so don't include them
    # --> might want to also confirm that this is bc off a selector not some oter reason
    if len(cols_to_replace) != len(X_t_single_col.columns):
        X_t_single_col = X_t_single_col[list(cols_to_replace)]

    X_t[list(cols_to_replace)] = X_t_single_col
    return X_t
