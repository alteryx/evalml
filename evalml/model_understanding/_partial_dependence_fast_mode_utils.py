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
    # Make sure that only components that are capable of handling fast mode are in the pipeline
    new_parameters = pipeline.parameters
    for component in pipeline.component_graph.component_instances.values():
        new_parameters = component._handle_partial_dependence_fast_mode(
            X,
            new_parameters,
        )

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
