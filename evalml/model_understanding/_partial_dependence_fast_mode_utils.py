import pandas as pd

from evalml.utils.woodwork_utils import infer_feature_types


def _get_cloned_feature_pipelines(
    features,
    pipeline,
    variable_has_features_passed_to_estimator,
    X_train,
    y_train,
):
    """Clones and fits pipelines for partial dependence fast mode.

    Args:
        pipeline (PipelineBase or subclass): Fitted pipeline that will be cloned.
        features (string, tuple[int or string]): The target feature for which to create the partial dependence plot for.
            If features is a string, it must be a valid column name in X.
            If features is a tuple of int/strings, it must contain valid column integers/names in X.
        X_train (pd.DataFrame, np.ndarray): The data that was used to train the original pipeline. Will
            be used to train the cloned pipelines.
        y_train (pd.Series, np.ndarray): The target data that was used to train the original pipeline. Will
            be used to train the cloned pipelines.

    Returns:
        dict[str, PipelineBase or subclass]: Dictionary mapping feature name to the pipeline pipeline
            fit for it.
    """
    if X_train is None or y_train is None:
        raise ValueError("Training data is required for partial dependence fast mode.")

    X_train = infer_feature_types(X_train)

    # Make sure that only components that are capable of handling fast mode are in the pipeline
    new_parameters = pipeline.parameters
    for component in pipeline.component_graph.component_instances.values():
        new_parameters = component._handle_partial_dependence_fast_mode(
            X_train,
            new_parameters,
        )

    # Create a fit pipeline for each feature
    cloned_feature_pipelines = {}
    for variable in features:
        # Don't fit pipelines if the feature has no impact on predictions
        if not variable_has_features_passed_to_estimator[variable]:
            continue
        pipeline_copy = pipeline.new(
            parameters=new_parameters,
        )
        pipeline_copy.fit(X_train.ww[[variable]], y_train)
        cloned_feature_pipelines[variable] = pipeline_copy

    return cloned_feature_pipelines


def _transform_single_feature(
    X,
    X_t,
    feature_provenance,
    variable,
    part_dep_column,
    cloned_pipeline,
):
    """Transforms single column using cloned pipeline and column that has been updated for partial dependence calculations.

    Args:
        X_t (pd.DataFrame, np.ndarray): The transformed data into which we insert the transformed single column.
        X (pd.DataFrame, np.ndarray): The original data used for calculating partial dependence.
        feature_provenance (dict[str, set[str]]): Dictionary mapping base feature names to engineered feature names.
        variable (str): The name of the single column we're transforming.
        part_dep_column (pd.Series): The updated column that will be transformed by the fit pipeline.
        cloned_pipeline (PipelineBase or subclass): Fitted pipeline that will be used to transform a single feature.

    Returns:
        dict[str, PipelineBase or subclass]: Dictionary mapping feature name to the pipeline pipeline
            fit for it.
    """
    changed_col_df = pd.DataFrame({variable: part_dep_column})
    changed_col_df.ww.init(
        logical_types={variable: X.ww.logical_types[variable]},
    )

    # Take the changed column and send it through transform by itself
    X_t_single_col = cloned_pipeline.transform_all_but_final(changed_col_df)

    # Determine which columns in X_t we're replacing with the transform output
    cols_to_replace = [variable]
    if feature_provenance.get(variable):
        # cols to replace has to be in the same order as X_t
        cols_to_replace = [
            col for col in X_t if col in feature_provenance.get(variable)
        ]
    # If some categories got dropped during transform of the original X_t,
    # they won't be in X_t_single_col, so don't include them
    if len(cols_to_replace) != len(X_t_single_col.columns):
        X_t_single_col = X_t_single_col[cols_to_replace]

    X_t[cols_to_replace] = X_t_single_col
    return X_t
