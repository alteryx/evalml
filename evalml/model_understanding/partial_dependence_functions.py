"""Top level functions for running partial dependence."""
import warnings

import numpy as np
import pandas as pd
import woodwork as ww

import evalml
from evalml.exceptions import (
    NullsInColumnWarning,
    PartialDependenceError,
    PartialDependenceErrorCode,
)
from evalml.model_family import ModelFamily
from evalml.model_understanding._partial_dependence_utils import (
    _add_ice_plot,
    _get_feature_names_from_str_or_col_index,
    _is_feature_of_semantic_type,
    _is_feature_of_type,
    _partial_dependence,
    _put_categorical_feature_first,
    _raise_value_error_if_any_features_all_nan,
    _raise_value_error_if_mostly_one_value,
    _range_for_dates,
)
from evalml.model_understanding.visualizations import _calculate_axis_range
from evalml.utils import import_or_raise, infer_feature_types, jupyter_check


def partial_dependence(
    pipeline,
    X,
    features,
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    kind="average",
    fast_mode=False,
    X_train=None,
    y_train=None,
):
    """Calculates one or two-way partial dependence.

    If a single integer or string is given for features, one-way partial dependence is calculated. If
    a tuple of two integers or strings is given, two-way partial dependence
    is calculated with the first feature in the y-axis and second feature in the x-axis.

    Args:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (pd.DataFrame, np.ndarray): The input data used to generate a grid of values
            for feature where partial dependence will be calculated at
        features (int, string, tuple[int or string]): The target feature for which to create the partial dependence plot for.
            If features is an int, it must be the index of the feature to use.
            If features is a string, it must be a valid column name in X.
            If features is a tuple of int/strings, it must contain valid column integers/names in X.
        percentiles (tuple[float]): The lower and upper percentile used to create the extreme values for the grid.
            Must be in [0, 1]. Defaults to (0.05, 0.95).
        grid_resolution (int): Number of samples of feature(s) for partial dependence plot.  If this value
            is less than the maximum number of categories present in categorical data within X, it will be
            set to the max number of categories + 1. Defaults to 100.
        kind ({'average', 'individual', 'both'}): The type of predictions to return. 'individual' will return the predictions for
            all of the points in the grid for each sample in X. 'average' will return the predictions for all of the points in
            the grid but averaged over all of the samples in X.
        fast_mode (bool, optional): Whether or not performance optimizations should be
            used for partial dependence calculations. Defaults to False.
            Note that user-specified components may not produce correct partial dependence results, so fast mode
            should only be used with EvalML-native components. Additionally, some components are not compatible
            with fast mode; in those cases, an error will be raised indicating that fast mode should not be used.
        X_train (pd.DataFrame, np.ndarray): The data that was used to train the original pipeline. Will
            be used in fast mode to train the cloned pipelines. Defaults to None.
        y_train (pd.Series, np.ndarray): The target data that was used to train the original pipeline. Will
            be used in fast mode to train the cloned pipelines. Defaults to None.

    Returns:
        pd.DataFrame, list(pd.DataFrame), or tuple(pd.DataFrame, list(pd.DataFrame)):
            When `kind='average'`: DataFrame with averaged predictions for all points in the grid averaged
            over all samples of X and the values used to calculate those predictions.

            When `kind='individual'`: DataFrame with individual predictions for all points in the grid for each sample
            of X and the values used to calculate those predictions. If a two-way partial dependence is calculated, then
            the result is a list of DataFrames with each DataFrame representing one sample's predictions.

            When `kind='both'`: A tuple consisting of the averaged predictions (in a DataFrame) over all samples of X and the individual
            predictions (in a list of DataFrames) for each sample of X.

            In the one-way case: The dataframe will contain two columns, "feature_values" (grid points at which the
            partial dependence was calculated) and "partial_dependence" (the partial dependence at that feature value).
            For classification problems, there will be a third column called "class_label" (the class label for which
            the partial dependence was calculated). For binary classification, the partial dependence is only calculated
            for the "positive" class.

            In the two-way case: The data frame will contain grid_resolution number of columns and rows where the
            index and column headers are the sampled values of the first and second features, respectively, used to make
            the partial dependence contour. The values of the data frame contain the partial dependence data for each
            feature value pair.

    Raises:
        ValueError: Error during call to scikit-learn's partial dependence method.
        Exception: All other errors during calculation.
        PartialDependenceError: if the user provides a tuple of not exactly two features.
        PartialDependenceError: if the provided pipeline isn't fitted.
        PartialDependenceError: if the provided pipeline is a Baseline pipeline.
        PartialDependenceError: if any of the features passed in are completely NaN
        PartialDependenceError: if any of the features are low-variance. Defined as having one value occurring more than the upper
            percentile passed by the user. By default 95%.
    """
    try:
        # Dynamically set the grid resolution to the maximum number of values
        # in the categorical/datetime variables if there are more categories/datetime values than resolution cells
        X = infer_feature_types(X)

        if isinstance(features, (list, tuple)):
            is_categorical = [
                _is_feature_of_semantic_type(f, X, "category") for f in features
            ]
            is_datetime = [
                _is_feature_of_type(f, X, ww.logical_types.Datetime) for f in features
            ]
        else:
            is_categorical = [_is_feature_of_semantic_type(features, X, "category")]
            is_datetime = [_is_feature_of_type(features, X, ww.logical_types.Datetime)]

        if isinstance(features, (list, tuple)):
            if any(is_datetime) and len(features) > 1:
                raise PartialDependenceError(
                    "Two-way partial dependence is not supported for datetime columns.",
                    code=PartialDependenceErrorCode.TWO_WAY_REQUESTED_FOR_DATES,
                )
            if len(features) != 2:
                raise PartialDependenceError(
                    "Too many features given to graph_partial_dependence.  Only one or two-way partial "
                    "dependence is supported.",
                    code=PartialDependenceErrorCode.TOO_MANY_FEATURES,
                )
            if not (
                all([isinstance(x, str) for x in features])
                or all([isinstance(x, int) for x in features])
            ):
                raise PartialDependenceError(
                    "Features provided must be a tuple entirely of integers or strings, not a mixture of both.",
                    code=PartialDependenceErrorCode.FEATURES_ARGUMENT_INCORRECT_TYPES,
                )
            feature_names = _get_feature_names_from_str_or_col_index(X, features)
        else:
            feature_names = _get_feature_names_from_str_or_col_index(X, [features])

        X_features = X.ww.loc[:, feature_names]
        X_unknown = X_features.ww.select("unknown")
        if len(X_unknown.columns):
            # We drop the unknown columns in the pipelines, so we cannot calculate partial dependence for these
            raise PartialDependenceError(
                f"Columns {X_unknown.columns.values} are of type 'Unknown', which cannot be used for partial dependence",
                code=PartialDependenceErrorCode.INVALID_FEATURE_TYPE,
            )

        X_not_allowed = X_features.ww.select(["URL", "EmailAddress", "NaturalLanguage"])
        if len(X_not_allowed.columns):
            # these three logical types aren't allowed for partial dependence
            types = sorted(
                set(X_not_allowed.ww.types["Logical Type"].astype(str).tolist()),
            )
            raise PartialDependenceError(
                f"Columns {X_not_allowed.columns.tolist()} are of types {types}, which cannot be used for partial dependence",
                code=PartialDependenceErrorCode.INVALID_FEATURE_TYPE,
            )

        if fast_mode and (X_train is None or y_train is None):
            raise ValueError(
                "Training data is required for partial dependence fast mode.",
            )

        X_cats = X_features.ww.select("category")
        X_dt = X_features.ww.select("datetime")

        if any(is_categorical):
            custom_range = {
                cat: list(X_cats[cat].dropna().unique()) for cat in X_cats.columns
            }
        elif any(is_datetime):
            custom_range = {
                date: _range_for_dates(
                    X_dt.ww.loc[:, date],
                    percentiles,
                    grid_resolution,
                )
                for date in X_dt.columns
            }
        else:
            custom_range = {}

        if not pipeline._is_fitted:
            raise PartialDependenceError(
                "Pipeline to calculate partial dependence for must be fitted",
                code=PartialDependenceErrorCode.UNFITTED_PIPELINE,
            )
        if pipeline.model_family == ModelFamily.BASELINE:
            raise PartialDependenceError(
                "Partial dependence plots are not supported for Baseline pipelines",
                code=PartialDependenceErrorCode.PIPELINE_IS_BASELINE,
            )

        feature_list = X[feature_names]
        _raise_value_error_if_any_features_all_nan(feature_list)

        if feature_list.isnull().sum().any():
            warnings.warn(
                "There are null values in the features, which will cause NaN values in the partial dependence output. "
                "Fill in these values to remove the NaN values.",
                NullsInColumnWarning,
            )

        _raise_value_error_if_mostly_one_value(feature_list, percentiles[1])

        try:
            preds = _partial_dependence(
                pipeline,
                X=X,
                features=feature_names,
                percentiles=percentiles,
                grid_resolution=grid_resolution,
                kind=kind,
                custom_range=custom_range,
                fast_mode=fast_mode,
                X_train=X_train,
                y_train=y_train,
            )
        except ValueError as e:
            if "percentiles are too close to each other" in str(e):
                raise PartialDependenceError(
                    "The scale of these features is too small and results in"
                    "percentiles that are too close together.  Partial dependence"
                    "cannot be computed for these types of features.  Consider"
                    "scaling the features so that they differ by > 10E-7",
                    code=PartialDependenceErrorCode.COMPUTED_PERCENTILES_TOO_CLOSE,
                )
            else:
                raise e

        classes = None
        if isinstance(pipeline, evalml.pipelines.BinaryClassificationPipeline):
            classes = [pipeline.classes_[1]]
        elif isinstance(pipeline, evalml.pipelines.MulticlassClassificationPipeline):
            classes = pipeline.classes_

        values = preds["values"]
        if kind in ["average", "both"]:
            avg_pred = preds["average"]
            if isinstance(features, (int, str)):
                avg_data = pd.DataFrame(
                    {
                        "feature_values": np.tile(values[0], avg_pred.shape[0]),
                        "partial_dependence": np.concatenate(
                            [pred for pred in avg_pred],
                        ),
                    },
                )
            elif isinstance(features, (list, tuple)):
                avg_data = pd.DataFrame(avg_pred.reshape((-1, avg_pred.shape[-1])))
                avg_data.columns = values[1]
                avg_data.index = np.tile(values[0], avg_pred.shape[0])

            if classes is not None:
                avg_data["class_label"] = np.repeat(classes, len(values[0]))

        if kind in ["individual", "both"]:
            ind_preds = preds["individual"]
            if isinstance(features, (int, str)):
                ind_data = list()
                for label in ind_preds:
                    ind_data.append(pd.DataFrame(label).T)

                ind_data = pd.concat(ind_data)
                ind_data.columns = [f"Sample {i}" for i in range(len(ind_preds[0]))]

                if classes is not None:
                    ind_data["class_label"] = np.repeat(classes, len(values[0]))
                ind_data.insert(
                    0,
                    "feature_values",
                    np.tile(values[0], ind_preds.shape[0]),
                )

            elif isinstance(features, (list, tuple)):
                ind_data = list()
                for n, label in enumerate(ind_preds):
                    for i, sample in enumerate(label):
                        ind_df = pd.DataFrame(sample.reshape((-1, sample.shape[-1])))
                        ind_df.columns = values[1]
                        ind_df.index = values[0]

                        if n == 0:
                            ind_data.append(ind_df)
                        else:
                            ind_data[i] = pd.concat([ind_data[i], ind_df])

                for sample in ind_data:
                    sample["class_label"] = np.repeat(classes, len(values[0]))

        if kind == "both":
            return (avg_data, ind_data)
        elif kind == "individual":
            return ind_data
        elif kind == "average":
            return avg_data
    except Exception as e:
        if isinstance(e, PartialDependenceError):
            raise e
        else:
            raise PartialDependenceError(
                str(e),
                PartialDependenceErrorCode.ALL_OTHER_ERRORS,
            ) from e


def graph_partial_dependence(
    pipeline,
    X,
    features,
    class_label=None,
    grid_resolution=100,
    kind="average",
):
    """Create an one-way or two-way partial dependence plot.

    Passing a single integer or string as features will create a one-way partial dependence plot with the feature values
    plotted against the partial dependence.  Passing features a tuple of int/strings will create
    a two-way partial dependence plot with a contour of feature[0] in the y-axis, feature[1]
    in the x-axis and the partial dependence in the z-axis.

    Args:
        pipeline (PipelineBase or subclass): Fitted pipeline.
        X (pd.DataFrame, np.ndarray): The input data used to generate a grid of values
            for feature where partial dependence will be calculated at.
        features (int, string, tuple[int or string]): The target feature for which to create the partial dependence plot for.
            If features is an int, it must be the index of the feature to use.
            If features is a string, it must be a valid column name in X.
            If features is a tuple of strings, it must contain valid column int/names in X.
        class_label (string, optional): Name of class to plot for multiclass problems. If None, will plot
            the partial dependence for each class. This argument does not change behavior for regression or binary
            classification pipelines. For binary classification, the partial dependence for the positive label will
            always be displayed. Defaults to None.
        grid_resolution (int): Number of samples of feature(s) for partial dependence plot.
        kind ({'average', 'individual', 'both'}): Type of partial dependence to plot. 'average' creates a regular partial dependence
             (PD) graph, 'individual' creates an individual conditional expectation (ICE) plot, and 'both' creates a
             single-figure PD and ICE plot. ICE plots can only be shown for one-way partial dependence plots.

    Returns:
        plotly.graph_objects.Figure: figure object containing the partial dependence data for plotting

    Raises:
        PartialDependenceError: if a graph is requested for a class name that isn't present in the pipeline.
        PartialDependenceError: if an ICE plot is requested for a two-way partial dependence.
    """
    X = infer_feature_types(X)
    if isinstance(features, (list, tuple)):
        mode = "two-way"
        is_categorical = [
            _is_feature_of_semantic_type(f, X, "category") for f in features
        ]
        if any(is_categorical):
            features = _put_categorical_feature_first(features, is_categorical[0])
        if kind == "individual" or kind == "both":
            raise PartialDependenceError(
                "Individual conditional expectation plot can only be created with a one-way partial dependence plot",
                PartialDependenceErrorCode.ICE_PLOT_REQUESTED_FOR_TWO_WAY_PLOT,
            )
    elif isinstance(features, (int, str)):
        mode = "one-way"
        is_categorical = _is_feature_of_semantic_type(features, X, "category")

    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)
    if (
        isinstance(pipeline, evalml.pipelines.MulticlassClassificationPipeline)
        and class_label is not None
    ):
        if class_label not in pipeline.classes_:
            msg = f"Class {class_label} is not one of the classes the pipeline was fit on: {', '.join(list(pipeline.classes_))}"
            raise PartialDependenceError(
                msg,
                code=PartialDependenceErrorCode.INVALID_CLASS_LABEL,
            )

    part_dep = partial_dependence(
        pipeline,
        X,
        features=features,
        grid_resolution=grid_resolution,
        kind=kind,
    )

    ice_data = None
    if kind == "both":
        part_dep, ice_data = part_dep
    elif kind == "individual":
        ice_data = part_dep
        part_dep = None

    if mode == "two-way":
        title = f"Partial Dependence of '{features[0]}' vs. '{features[1]}'"
        layout = _go.Layout(
            title={"text": title},
            xaxis={"title": f"{features[1]}"},
            yaxis={"title": f"{features[0]}"},
            showlegend=True,
        )
    elif mode == "one-way":
        feature_name = str(features)
        if kind == "individual":
            title = f"Individual Conditional Expectation of '{feature_name}'"
        elif kind == "average":
            title = f"Partial Dependence of '{feature_name}'"
        else:
            title = f"Partial Dependence of '{feature_name}' <br><sub>Including Individual Conditional Expectation Plot</sub>"
        layout = _go.Layout(
            title={"text": title},
            xaxis={"title": f"{feature_name}"},
            yaxis={"title": "Partial Dependence"},
            showlegend=True,
        )

    fig = _go.Figure(layout=layout)
    if isinstance(pipeline, evalml.pipelines.MulticlassClassificationPipeline):
        class_labels = [class_label] if class_label is not None else pipeline.classes_
        _subplots = import_or_raise(
            "plotly.subplots",
            error_msg="Cannot find dependency plotly.graph_objects",
        )

        # If the user passes in a value for class_label, we want to create a 1 x 1 subplot or else there would
        # be an empty column in the plot and it would look awkward
        rows, cols = (
            ((len(class_labels) + 1) // 2, 2)
            if len(class_labels) > 1
            else (1, len(class_labels))
        )

        class_labels_mapping = {
            class_label: str(class_label) for class_label in class_labels
        }
        # Don't specify share_xaxis and share_yaxis so that we get tickmarks in each subplot
        fig = _subplots.make_subplots(rows=rows, cols=cols, subplot_titles=class_labels)
        for i, label in enumerate(class_labels):
            label_df = (
                part_dep.loc[part_dep.class_label == label]
                if part_dep is not None
                else ice_data.loc[ice_data.class_label == label]
            )
            row = (i + 2) // 2
            col = (i % 2) + 1
            if ice_data is not None and kind == "individual":
                fig = _add_ice_plot(_go, fig, ice_data, row=row, col=col, label=label)
            else:
                label_df.drop(columns=["class_label"], inplace=True)
                if mode == "two-way":
                    _update_fig_with_two_way_partial_dependence(
                        _go,
                        fig,
                        label_df,
                        part_dep,
                        features,
                        is_categorical,
                        label,
                        row,
                        col,
                    )
                elif mode == "one-way":
                    x = label_df["feature_values"]
                    y = label_df["partial_dependence"]
                    if is_categorical:
                        trace = _go.Bar(x=x, y=y, name=label)
                    else:
                        if ice_data is not None:
                            fig = _add_ice_plot(
                                _go,
                                fig,
                                ice_data,
                                row=row,
                                col=col,
                                label=label,
                            )
                        trace = _go.Scatter(
                            x=x,
                            y=y,
                            line=dict(width=3, color="rgb(99,110,250)"),
                            name="Partial Dependence: " + class_labels_mapping[label],
                        )
                    fig.add_trace(trace, row=row, col=col)

        fig.update_layout(layout)

        if mode == "two-way":
            fig.update_layout(coloraxis=dict(colorscale="Bluered_r"), showlegend=False)
        elif mode == "one-way":
            title = f"{feature_name}"
            x_scale_df = (
                part_dep["feature_values"]
                if part_dep is not None
                else ice_data["feature_values"]
            )
            xrange = _calculate_axis_range(x_scale_df) if not is_categorical else None
            yrange = _calculate_axis_range(
                ice_data.drop("class_label", axis=1)
                if ice_data is not None
                else part_dep["partial_dependence"],
            )
            fig.update_xaxes(title=title, range=xrange)
            fig.update_yaxes(range=yrange)
    elif kind == "individual" and ice_data is not None:
        fig = _add_ice_plot(_go, fig, ice_data)
    elif part_dep is not None:
        if ice_data is not None and not is_categorical:
            fig = _add_ice_plot(_go, fig, ice_data)
        if "class_label" in part_dep.columns:
            part_dep.drop(columns=["class_label"], inplace=True)
        if mode == "two-way":
            _update_fig_with_two_way_partial_dependence(
                _go,
                fig,
                part_dep,
                part_dep,
                features,
                is_categorical,
                label="Partial Dependence",
                row=None,
                col=None,
            )
        elif mode == "one-way":
            if is_categorical:
                trace = _go.Bar(
                    x=part_dep["feature_values"],
                    y=part_dep["partial_dependence"],
                    name="Partial Dependence",
                )
            else:
                trace = _go.Scatter(
                    x=part_dep["feature_values"],
                    y=part_dep["partial_dependence"],
                    name="Partial Dependence",
                    line=dict(width=3, color="rgb(99,110,250)"),
                )
            fig.add_trace(trace)
    return fig


def _update_fig_with_two_way_partial_dependence(
    _go,
    fig,
    label_df,
    part_dep,
    features,
    is_categorical,
    label=None,
    row=None,
    col=None,
):
    """Helper for formatting the two-way partial dependence plot."""
    y = label_df.index
    x = label_df.columns
    z = label_df.values
    if not any(is_categorical):
        # No features are categorical. In this case, we pass both x and y data to the Contour plot so that
        # plotly can figure out the axis formatting for us.
        kwargs = {"x": x, "y": y}
        fig.update_xaxes(
            title=f"{features[1]}",
            range=_calculate_axis_range(
                np.array([x for x in part_dep.columns if x != "class_label"]),
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(range=_calculate_axis_range(part_dep.index), row=row, col=col)
    elif sum(is_categorical) == 1:
        # One feature is categorical. Since we put the categorical feature first, the numeric feature will be the x
        # axis. So we pass the x to the Contour plot so that plotly can format it for us.
        # Since the y axis is a categorical value, we will set the y tickmarks ourselves. Passing y to the contour plot
        # in this case will "work" but the formatting will look bad.
        kwargs = {"x": x}
        fig.update_xaxes(
            title=f"{features[1]}",
            range=_calculate_axis_range(
                np.array([x for x in part_dep.columns if x != "class_label"]),
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(label_df.shape[0])),
            ticktext=list(label_df.index),
            row=row,
            col=col,
        )
    else:
        # Both features are categorical so we must format both axes ourselves.
        kwargs = {}
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(label_df.shape[0])),
            ticktext=list(label_df.index),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(label_df.shape[1])),
            ticktext=list(label_df.columns),
            row=row,
            col=col,
        )
    fig.add_trace(
        _go.Contour(z=z, name=label, coloraxis="coloraxis", **kwargs),
        row=row,
        col=col,
    )
