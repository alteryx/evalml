"""Partial dependence implementation and utility functions.

Implementation borrows from sklearn "brute" calculation but with our
own modification to better handle mixed data types in the grid
as well as EvalML pipelines.
"""
import numpy as np
import pandas as pd
import woodwork as ww
from scipy.stats.mstats import mquantiles

from evalml.exceptions import PartialDependenceError, PartialDependenceErrorCode
from evalml.model_understanding._partial_dependence_fast_mode_utils import (
    _get_cloned_feature_pipelines,
    _transform_single_feature,
)
from evalml.problem_types import is_regression


def _add_ice_plot(_go, fig, ice_data, label=None, row=None, col=None):
    x = ice_data["feature_values"]
    y = ice_data
    if "class_label" in ice_data.columns:
        if label:
            y = y[y["class_label"] == label]
        y.drop(columns=["class_label"], inplace=True)
    y = y.drop(columns=["feature_values"])
    for i, sample in enumerate(y):
        fig.add_trace(
            _go.Scatter(
                x=x,
                y=y[sample],
                line=dict(width=0.5, color="gray"),
                name=f"Individual Conditional Expectation{': ' + label if label else ''}",
                legendgroup="ICE" + label if label else "ICE",
                showlegend=True if i == 0 else False,
            ),
            row=row,
            col=col,
        )
    return fig


def _is_feature_of_type(feature, X, ltype):
    """Determine whether the feature the user passed in to partial dependence is a Woodwork logical type."""
    if isinstance(feature, int):
        is_type = isinstance(X.ww.logical_types[X.columns[feature]], ltype)
    else:
        is_type = isinstance(X.ww.logical_types[feature], ltype)
    return is_type


def _is_feature_of_semantic_type(feature, X, stype):
    """Determine whether the feature the user passed to partial dependence is a certain Woodwork semantic type."""
    if isinstance(feature, int):
        is_type = stype in X.ww.semantic_tags[X.columns[feature]]
    else:
        is_type = stype in X.ww.semantic_tags[feature]
    return is_type


def _put_categorical_feature_first(features, first_feature_categorical):
    """If the user is doing a two-way partial dependence plot and one of the features is categorical, we need to ensure the categorical feature is the first element in the tuple that's passed to sklearn.

    This is because in the two-way grid calculation, sklearn will try to coerce every element of the grid to the
    type of the first feature in the tuple. If we put the categorical feature first, the grid will be of type 'object'
    which can accommodate both categorical and numeric data. If we put the numeric feature first, the grid will be of
    type float64 and we can't coerce categoricals to float64 dtype.
    """
    new_features = features if first_feature_categorical else (features[1], features[0])
    return new_features


def _get_feature_names_from_str_or_col_index(X, names_or_col_indices):
    """Helper function to map the user-input features param to column names."""
    feature_list = []
    for name_or_index in names_or_col_indices:
        if isinstance(name_or_index, int):
            feature_list.append(X.columns[name_or_index])
        else:
            feature_list.append(name_or_index)
    return feature_list


def _raise_value_error_if_any_features_all_nan(df):
    """Helper for partial dependence data validation."""
    nan_pct = df.isna().mean()
    all_nan = nan_pct[nan_pct == 1].index.tolist()
    all_nan = [f"'{name}'" for name in all_nan]

    if all_nan:
        raise PartialDependenceError(
            "The following features have all NaN values and so the "
            f"partial dependence cannot be computed: {', '.join(all_nan)}",
            PartialDependenceErrorCode.FEATURE_IS_ALL_NANS,
        )


def _raise_value_error_if_mostly_one_value(df, percentile):
    """Helper for partial dependence data validation."""
    one_value = []
    values = []

    for col in df.columns:
        normalized_counts = df[col].value_counts(normalize=True) + 0.01
        normalized_counts = normalized_counts[normalized_counts > percentile]
        if not normalized_counts.empty:
            one_value.append(f"'{col}'")
            values.append(str(normalized_counts.index[0]))

    if one_value:
        raise PartialDependenceError(
            f"Features ({', '.join(one_value)}) are mostly one value, ({', '.join(values)}), "
            f"and cannot be used to compute partial dependence. Try raising the upper percentage value.",
            PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE,
        )


def _range_for_dates(X_dt, percentiles, grid_resolution):
    """Compute the range of values used in partial dependence for datetime features.

    Interpolate between the percentiles of the dates converted to unix
    timestamps.

    Args:
        X_dt (pd.DataFrame): Datetime features in original data. We currently
            only support X_dt having a single column.
        percentiles (tuple float): Percentiles to interpolate between.
        grid_resolution (int): Number of points in range.

    Returns:
        pd.Series: Range of dates between percentiles.
    """
    timestamps = np.array(
        [X_dt - pd.Timestamp("1970-01-01")] // np.timedelta64(1, "s"),
    ).reshape(-1, 1)
    timestamps = pd.DataFrame(timestamps)
    grid, values = _grid_from_X(
        timestamps,
        percentiles=percentiles,
        grid_resolution=grid_resolution,
        custom_range={},
    )
    grid_dates = pd.to_datetime(pd.Series(grid.squeeze()), unit="s")
    return grid_dates


def _grid_from_X(X, percentiles, grid_resolution, custom_range):
    """Create cartesian product of all the columns of input dataframe X.

    Args:
        X (pd.DataFrame): Input data
        percentiles (tuple float): Percentiles to use as endpoints of the grid
            for each feature.
        grid_resolution (int): How many points to interpolate between percentiles.
        custom_range (dict[str, np.ndarray]): Mapping from column name in X to
            range of values to use in partial dependence. If custom_range is specified,
            the percentile + interpolation procedure is skipped and the values in custom_range
            are used.

    Returns:
        pd.DataFrame: Cartesian product of input columns of X.
    """
    values = []
    for feature in X.columns:
        if feature in custom_range:
            # Use values in the custom range
            feature_range = custom_range[feature]
            if not isinstance(feature_range, (np.ndarray, pd.Series)):
                feature_range = np.array(feature_range)
            if feature_range.ndim != 1:
                raise ValueError(
                    "Grid for feature {} is not a one-dimensional array. Got {}"
                    " dimensions".format(feature, feature_range.ndim),
                )
            axis = feature_range
        else:
            feature_vector = X.loc[:, feature].dropna()
            uniques = np.unique(feature_vector)
            if uniques.shape[0] < grid_resolution:
                # feature has low resolution use unique vals
                axis = uniques
            else:
                # create axis based on percentiles and grid resolution
                emp_percentiles = mquantiles(feature_vector, prob=percentiles, axis=0)
                if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                    raise ValueError(
                        "percentiles are too close to each other, "
                        "unable to build the grid. Please choose percentiles "
                        "that are further apart.",
                    )
                axis = np.linspace(
                    emp_percentiles[0],
                    emp_percentiles[1],
                    num=grid_resolution,
                    endpoint=True,
                )
        values.append(axis)

    return _cartesian(values), values


def _cartesian(arrays):
    """Create cartesian product of elements of arrays list.

    Stored in a dataframe to allow mixed types like dates/str/numeric.

    Args:
        arrays (list(np.ndarray)): Arrays.

    Returns:
        pd.DataFrame: Cartesian product of arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = pd.DataFrame()

    for n, arr in enumerate(arrays):
        out[n] = arrays[n][ix[:, n]]

    return out


def _partial_dependence_calculation(
    pipeline,
    grid,
    features,
    X,
    X_train,
    y_train,
    fast_mode=False,
):
    """Do the partial dependence calculation once the grid is computed.

    Args:
        pipeline (PipelineBase): pipeline.
        grid (pd.DataFrame): Grid of features to compute the partial dependence on.
        features (list(str)): Column names of input data
        X (pd.DataFrame): Input data.
        fast_mode (bool, optional): Whether or not performance optimizations should be
            used. Defaults to False. When True, copies of pipelines will be used to transform just the
            column(s) we're calculating partial dependence for. This means that any pipeline
            containing a component that relies on multiple columns for fit and transform should
            not be used. See the ``_can_be_used_for_fast_partial_dependence`` property on components
            to determine which components cannot be used for fast mode.
        X_train (pd.DataFrame, np.ndarray): The data that was used to train the original pipeline. Will
            be used in fast mode to train the cloned pipelines.
        y_train (pd.Series, np.ndarray): The target data that was used to train the original pipeline. Will
            be used in fast mode to train the cloned pipelines.

    Returns:
        Tuple (np.ndarray, np.ndarray): averaged and individual predictions for
            all points in the grid.
    """
    X_eval = X.ww.copy()
    prediction_object = pipeline
    if fast_mode:
        # In fast mode, we alter the transformed X with our partial dependence values
        # and then call predict directly with the estimator
        X_eval = pipeline.transform_all_but_final(X_eval)
        prediction_object = pipeline.estimator

        # Some components may drop features, so we need to know whether the specified
        # features actually have an impact on predictions
        feature_provenance = pipeline._get_feature_provenance()
        variable_has_features_passed_to_estimator = {
            variable: variable in feature_provenance or variable in X_eval.columns
            for variable in features
        }
        no_features_passed_to_estimator = not any(
            variable_has_features_passed_to_estimator.values(),
        )

        # Fit pipelines for each feature so that we can transform it with grid
        # values later
        cloned_feature_pipelines = _get_cloned_feature_pipelines(
            features,
            pipeline,
            variable_has_features_passed_to_estimator,
            X_train,
            y_train,
        )

    if is_regression(pipeline.problem_type):
        prediction_method = prediction_object.predict
    else:
        prediction_method = prediction_object.predict_proba
    if fast_mode and no_features_passed_to_estimator:
        original_predictions = prediction_method(X_eval)
        original_predictions_mean = np.mean(original_predictions, axis=0)

        predictions = [original_predictions for _, _ in grid.iterrows()]
        averaged_predictions = [original_predictions_mean for _, _ in grid.iterrows()]
    else:
        predictions = []
        averaged_predictions = []
        for _, new_values in grid.iterrows():
            for i, variable in enumerate(features):
                part_dep_column = pd.Series(
                    [new_values[i]] * X_eval.shape[0],
                    index=X_eval.index,
                )

                if fast_mode:
                    X_eval = _transform_single_feature(
                        X,
                        X_eval,
                        feature_provenance,
                        variable,
                        part_dep_column,
                        cloned_feature_pipelines[variable],
                    )
                else:
                    X_eval.ww[variable] = ww.init_series(
                        part_dep_column,
                        logical_type=X_eval.ww.logical_types[variable],
                        origin=X_eval.ww.columns[variable].origin,
                    )
            pred = prediction_method(X_eval)
            predictions.append(pred)
            # average over samples
            averaged_predictions.append(np.mean(pred, axis=0))
    n_samples = X.shape[0]

    # reshape to (n_instances, n_points) for binary/regression
    # reshape to (n_classes, n_instances, n_points) for multiclass
    predictions = np.array(predictions).T
    if is_regression(pipeline.problem_type) and predictions.ndim == 2:
        predictions = predictions.reshape(n_samples, -1)
    elif predictions.shape[0] == 2:
        predictions = predictions[1]
        predictions = predictions.reshape(n_samples, -1)

    # reshape averaged_predictions to (1, n_points) for binary/regression
    # reshape averaged_predictions to (n_classes, n_points) for multiclass.
    averaged_predictions = np.array(averaged_predictions).T
    if is_regression(pipeline.problem_type) and averaged_predictions.ndim == 1:
        averaged_predictions = averaged_predictions.reshape(1, -1)
    elif averaged_predictions.shape[0] == 2:
        averaged_predictions = averaged_predictions[1]
        averaged_predictions = averaged_predictions.reshape(1, -1)

    return averaged_predictions, predictions


def _partial_dependence(
    pipeline,
    X,
    features,
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    kind="average",
    custom_range=None,
    fast_mode=False,
    X_train=None,
    y_train=None,
):
    """Compute the partial dependence for features of X.

    Args:
        pipeline (PipelineBase): pipeline.
        X (pd.DataFrame): Holdout data
        features (list(str)): Column names of X to compute the partial dependence for.
        percentiles (tuple float): Percentiles to use in range calculation for a given
            feature.
        grid_resolution: Number of points in range of values used for each feature in
            partial dependence calculation.
        kind (str): The type of predictions to return.
        custom_range (dict[str, np.ndarray]): Mapping from column name in X to
            range of values to use in partial dependence. If custom_range is specified,
            the percentile + interpolation procedure is skipped and the values in custom_range
            are used.
        fast_mode (bool, optional): Whether or not performance optimizations should be
            used. Defaults to False. When True, copies of pipelines will be used to transform just the
            column(s) we're calculating partial dependence for. This means that any pipeline
            containing a component that relies on multiple columns for fit and transform should
            not be used. See the ``_can_be_used_for_fast_partial_dependence`` property on components
            to determine which components cannot be used for fast mode.
        X_train (pd.DataFrame, np.ndarray): The data that was used to train the original pipeline. Will
            be used in fast mode to train the cloned pipelines. Defaults to None.
        y_train (pd.Series, np.ndarray): The target data that was used to train the original pipeline. Will
            be used in fast mode to train the cloned pipelines. Defaults to None.

    Returns:
        dict with 'average', 'individual', 'values' keys. 'values' is a list of
            the values used in the partial dependence for each feature.
            'average' and 'individual' are averaged and individual predictions for
            each point in the grid.
    """
    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")

    custom_range = custom_range or {}
    custom_range = {
        feature: custom_range.get(feature)
        for feature in features
        if feature in custom_range
    }
    grid, values = _grid_from_X(
        X.loc[:, features],
        percentiles,
        grid_resolution,
        custom_range,
    )

    averaged_predictions, predictions = _partial_dependence_calculation(
        pipeline,
        grid,
        features,
        X,
        X_train=X_train,
        y_train=y_train,
        fast_mode=fast_mode,
    )

    # reshape predictions to
    # (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
    predictions = predictions.reshape(-1, X.shape[0], *[val.shape[0] for val in values])

    # reshape averaged_predictions to
    # (n_outputs, n_values_feature_0, n_values_feature_1, ...)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values]
    )

    if kind == "average":
        return {"average": averaged_predictions, "values": values}
    elif kind == "individual":
        return {"individual": predictions, "values": values}
    else:  # kind='both'
        return {
            "average": averaged_predictions,
            "individual": predictions,
            "values": values,
        }
