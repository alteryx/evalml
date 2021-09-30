"""Partial dependence implementation.

Borrows from sklearn "brute" calculation but with our
own modification to better handle mixed data types in the grid
as well as EvalML pipelines.
"""
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles

from evalml.problem_types import is_regression


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
        [X_dt - pd.Timestamp("1970-01-01")] // np.timedelta64(1, "s")
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
                    " dimensions".format(feature, feature_range.ndim)
                )
            axis = feature_range
        else:
            uniques = np.unique(X.loc[:, feature])
            if uniques.shape[0] < grid_resolution:
                # feature has low resolution use unique vals
                axis = uniques
            else:
                # create axis based on percentiles and grid resolution
                emp_percentiles = mquantiles(
                    X.loc[:, feature], prob=percentiles, axis=0
                )
                if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                    raise ValueError(
                        "percentiles are too close to each other, "
                        "unable to build the grid. Please choose percentiles "
                        "that are further apart."
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


def _partial_dependence_calculation(pipeline, grid, features, X):
    """Do the partial dependence calculation once the grid is computed.

    Args:
        pipeline (PipelineBase): pipeline.
        grid (pd.DataFrame): Grid of features to compute the partial dependence on.
        features (list(str)): Column names of input data
        X (pd.DataFrame): Input data.

    Returns:
        Tuple (np.ndarray, np.ndarray): averaged and individual predictions for
            all points in the grid.
    """
    predictions = []
    averaged_predictions = []

    if is_regression(pipeline.problem_type):
        prediction_method = pipeline.predict
    else:
        prediction_method = pipeline.predict_proba

    X_eval = X.copy()
    for _, new_values in grid.iterrows():
        for i, variable in enumerate(features):
            X_eval.loc[:, variable] = new_values[i]

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
