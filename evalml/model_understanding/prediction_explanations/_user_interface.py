import abc

import pandas as pd
from texttable import Texttable

from evalml.model_understanding.prediction_explanations._algorithms import (
    _aggregate_explainer_values,
    _compute_lime_values,
    _compute_shap_values,
    _normalize_explainer_values,
)
from evalml.problem_types import ProblemTypes


def _make_rows(
    explainer_values,
    normalized_values,
    pipeline_features,
    original_features,
    top_k,
    include_explainer_values=False,
    convert_numeric_to_string=True,
):
    """Makes the rows (one row for each feature) for the explanation table.

    Args:
        explainer_values (dict): Dictionary mapping the feature names to their explainer values. In a multiclass setting,
            this dictionary for correspond to the explainer values for a single class.
        normalized_values (dict): Normalized explainer values. Same structure as explainer_values parameter.
        pipeline_features (pd.Series): The features created by the pipeline.
        original_features (pd.Series): The features passed to the pipeline by the user. If possible,
            will display the original feature value.
        top_k (int): How many of the highest/lowest features to include in the table.
        include_explainer_values (bool): Whether to include the explainer values in their own column.
        convert_numeric_to_string (bool): Whether numeric values should be converted to strings from numeric

    Returns:
          list[str]
    """
    tuples = [
        (value[0], feature_name) for feature_name, value in normalized_values.items()
    ]

    # Sort the features s.t the top_k_features w the largest explainer value magnitudes are the first
    # top_k_features elements
    tuples = sorted(tuples, key=lambda x: abs(x[0]), reverse=True)

    # Then sort such that the explainer values go from most positive to most negative
    features_to_display = reversed(sorted(tuples[:top_k]))

    rows = []
    for value, feature_name in features_to_display:
        symbol = "+" if value >= 0 else "-"
        display_text = symbol * min(int(abs(value) // 0.2) + 1, 5)

        # At this point, the feature is either in the original data or the data
        # the final estimator sees, or both. We use the original feature value if possible
        is_original_feature = feature_name in original_features.columns
        if is_original_feature:
            feature_value = original_features[feature_name].iloc[0]
        else:
            feature_value = pipeline_features[feature_name].iloc[0]

        if convert_numeric_to_string:
            if pd.api.types.is_number(feature_value) and not pd.api.types.is_bool(
                feature_value,
            ):
                feature_value = "{:.2f}".format(feature_value)
            else:
                feature_value = str(feature_value)

        feature_value = _make_json_serializable(feature_value)

        row = [feature_name, feature_value, display_text]
        if include_explainer_values:
            explainer_value = explainer_values[feature_name][0]
            if convert_numeric_to_string:
                explainer_value = "{:.2f}".format(explainer_value)
            row.append(explainer_value)
        rows.append(row)

    return rows


def _rows_to_dict(rows):
    """Turns a list of lists into a dictionary."""
    feature_names = []
    feature_values = []
    qualitative_explanations = []
    quantitative_explanations = []
    for row in rows:
        name, value, qualitative = row[:3]
        quantitative = None
        if len(row) == 4:
            quantitative = row[-1]
        feature_names.append(name)
        feature_values.append(value)
        qualitative_explanations.append(qualitative)
        quantitative_explanations.append(quantitative)

    return {
        "feature_names": feature_names,
        "feature_values": feature_values,
        "qualitative_explanation": qualitative_explanations,
        "quantitative_explanation": quantitative_explanations,
    }


def _make_json_serializable(value):
    """Make sure a numeric boolean type is json serializable.

    numpy.int64 or numpy.bool can't be serialized to json.
    """
    # Base boolean are identified as numbers by pandas
    # so put the `is_bool` check prior
    if pd.api.types.is_bool(value):
        value = bool(value)
    elif pd.api.types.is_number(value):
        if pd.api.types.is_integer(value):
            value = int(value)
        else:
            value = float(value)
    elif isinstance(value, pd.Timestamp):
        value = str(value)

    return value


def _make_text_table(
    explainer_values,
    normalized_values,
    pipeline_features,
    original_features,
    top_k,
    include_explainer_values=False,
    algorithm="shap",
):
    """Make a table displaying the explainer values for a prediction.

    Args:
        explainer_values (dict): Dictionary mapping the feature names to their explainer values. In a multiclass setting,
            this dictionary for correspond to the explainer values for a single class.
        normalized_values (dict): Normalized explainer values. Same structure as explainer_values parameter.
        top_k (int): How many of the highest/lowest features to include in the table.
        include_explainer_values (bool): Whether to include the explainer values in their own column.

    Returns:
        str
    """
    n_cols = 4 if include_explainer_values else 3
    dtypes = ["t"] * n_cols
    alignment = ["c"] * n_cols

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(dtypes)
    table.set_cols_align(alignment)

    header = ["Feature Name", "Feature Value", "Contribution to Prediction"]
    if include_explainer_values:
        header.append(f"{algorithm.upper()} Value")

    rows = [header]
    rows += _make_rows(
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        top_k,
        include_explainer_values,
    )
    table.add_rows(rows)
    return table.draw()


class _TableMaker(abc.ABC):
    """Makes an explanation table for a regression, binary, or multiclass classification problem."""

    def __init__(
        self,
        top_k,
        include_explainer_values,
        include_expected_value,
        provenance,
        algorithm="shap",
    ):
        self.top_k = top_k
        self.include_explainer_values = include_explainer_values
        self.include_expected_value = include_expected_value
        self.provenance = provenance
        self.algorithm = algorithm

    @staticmethod
    def make_drill_down_dict(
        provenance,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        include_explainer_values,
    ):
        """Format the 'drill_down' section of the explanation report when output_format="dict". This section will include the feature values, feature names, qualitative explanation and explainer values (if include_explainer_values=True) for the features created from one of the original features in the data."""
        drill_down = {}
        for parent_feature, children_features in provenance.items():
            explainer_for_children = {
                k: v for k, v in explainer_values.items() if k in children_features
            }
            agg_for_children = {
                k: v for k, v in normalized_values.items() if k in children_features
            }
            top_k = len(agg_for_children)
            rows = _make_rows(
                explainer_for_children,
                agg_for_children,
                pipeline_features,
                original_features,
                top_k=top_k,
                include_explainer_values=include_explainer_values,
                convert_numeric_to_string=False,
            )
            drill_down[parent_feature] = _rows_to_dict(rows)
        return drill_down

    @abc.abstractmethod
    def make_text(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        """Creates a table given explainer values and formats it as text."""

    @abc.abstractmethod
    def make_dict(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        """Creates a table given explainer values and formats it as dictionary."""

    def make_dataframe(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        data = self.make_dict(
            aggregated_explainer_values,
            aggregated_normalized_values,
            explainer_values=explainer_values,
            normalized_values=normalized_values,
            pipeline_features=pipeline_features,
            original_features=original_features,
            expected_value=expected_value,
        )["explanations"]

        # Not including the drill down dict for dataframes
        # 'drill_down' is always included in the dict output so we can delete it
        for d in data:
            del d["drill_down"]
            del d["expected_value"]

        df = pd.concat(map(pd.DataFrame, data)).reset_index(drop=True)
        if "class_name" in df.columns and df["class_name"].isna().all():
            df = df.drop(columns=["class_name"])
        return df


class _RegressionExplanationTable(_TableMaker):
    """Makes an explanation table explaining a prediction for a regression problems."""

    def make_text(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        return _make_text_table(
            aggregated_explainer_values,
            aggregated_normalized_values,
            pipeline_features,
            original_features,
            self.top_k,
            self.include_explainer_values,
            algorithm=self.algorithm,
        )

    def make_dict(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        rows = _make_rows(
            aggregated_explainer_values,
            aggregated_normalized_values,
            pipeline_features,
            original_features,
            self.top_k,
            self.include_explainer_values,
            convert_numeric_to_string=False,
        )
        json_rows = _rows_to_dict(rows)
        drill_down = self.make_drill_down_dict(
            self.provenance,
            explainer_values,
            normalized_values,
            pipeline_features,
            original_features,
            self.include_explainer_values,
        )
        json_rows["class_name"] = None
        json_rows["drill_down"] = drill_down
        json_rows["expected_value"] = expected_value
        return {"explanations": [json_rows]}


class _BinaryExplanationTable(_TableMaker):
    """Makes an explanation table explaining a prediction for a binary classification problem."""

    def __init__(
        self,
        top_k,
        include_explainer_values,
        include_expected_value,
        class_names,
        provenance,
        algorithm="shap",
    ):
        super().__init__(
            top_k,
            include_explainer_values,
            include_expected_value,
            provenance,
            algorithm,
        )
        self.class_names = class_names

    def make_text(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        # The SHAP algorithm will return a two-element list for binary problems.
        # By convention, we display the explanation for the dominant class.
        return _make_text_table(
            aggregated_explainer_values[1],
            aggregated_normalized_values[1],
            pipeline_features,
            original_features,
            self.top_k,
            self.include_explainer_values,
            algorithm=self.algorithm,
        )

    def make_dict(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        rows = _make_rows(
            aggregated_explainer_values[1],
            aggregated_normalized_values[1],
            pipeline_features,
            original_features,
            self.top_k,
            self.include_explainer_values,
            convert_numeric_to_string=False,
        )
        dict_rows = _rows_to_dict(rows)
        drill_down = self.make_drill_down_dict(
            self.provenance,
            explainer_values[1],
            normalized_values[1],
            pipeline_features,
            original_features,
            self.include_explainer_values,
        )
        dict_rows["drill_down"] = drill_down
        dict_rows["class_name"] = _make_json_serializable(self.class_names[1])
        dict_rows["expected_value"] = expected_value

        return {"explanations": [dict_rows]}


class _MultiClassExplanationTable(_TableMaker):
    """Makes an exlpanation table explaining a prediction for a multiclass classification problem."""

    def __init__(
        self,
        top_k,
        include_explainer_values,
        include_expected_value,
        class_names,
        provenance,
        algorithm="shap",
    ):
        super().__init__(
            top_k,
            include_explainer_values,
            include_expected_value,
            provenance,
            algorithm,
        )
        self.class_names = class_names

    def make_text(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        strings = []
        for class_name, class_values, normalized_class_values in zip(
            self.class_names,
            aggregated_explainer_values,
            aggregated_normalized_values,
        ):
            strings.append(f"Class: {class_name}\n")
            table = _make_text_table(
                class_values,
                normalized_class_values,
                pipeline_features,
                original_features,
                self.top_k,
                self.include_explainer_values,
                algorithm=self.algorithm,
            )
            strings += table.splitlines()
            strings.append("\n")
        return "\n".join(strings)

    def make_dict(
        self,
        aggregated_explainer_values,
        aggregated_normalized_values,
        explainer_values,
        normalized_values,
        pipeline_features,
        original_features,
        expected_value,
    ):
        json_output = []
        for class_index, class_name in enumerate(self.class_names):
            rows = _make_rows(
                aggregated_explainer_values[class_index],
                aggregated_normalized_values[class_index],
                pipeline_features,
                original_features,
                self.top_k,
                self.include_explainer_values,
                convert_numeric_to_string=False,
            )
            json_output_for_class = _rows_to_dict(rows)
            drill_down = self.make_drill_down_dict(
                self.provenance,
                explainer_values[class_index],
                normalized_values[class_index],
                pipeline_features,
                original_features,
                self.include_explainer_values,
            )
            json_output_for_class["drill_down"] = drill_down
            expected_value_class = (
                expected_value[class_index] if expected_value is not None else None
            )
            json_output_for_class["expected_value"] = expected_value_class
            json_output_for_class["class_name"] = _make_json_serializable(class_name)
            json_output.append(json_output_for_class)
        return {"explanations": json_output}


def _make_single_prediction_explanation_table(
    pipeline,
    pipeline_features,
    input_features,
    index_to_explain,
    top_k=3,
    include_explainer_values=False,
    include_expected_value=False,
    output_format="text",
    algorithm="shap",
):
    """Creates table summarizing the top_k_features positive and top_k_features negative contributing features to the prediction of a single datapoint.

    Args:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP or LIME.
        pipeline_features (pd.DataFrame): Dataframe of features computed by the pipeline.
        input_features (pd.DataFrame): Dataframe of features passed to the pipeline. This is where the pipeline_features
            come from.
        index_to_explain (int): Index in the pipeline_features/input_features to explain.
        top_k (int): How many of the highest/lowest features to include in the table.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
        include_explainer_values (bool): Whether the explainer values should be included in an extra column in the output.
            Default is False.
        include_expected_value (bool): Whether the expected value should be included in the table. Default is False.
        output_format (str): The desired format of the output.  Can be "text", "dict", or "dataframe".
        algorithm (str): Algorithm to use while generating top contributing features, one of "shap" or "lime". Defaults to "shap".

    Returns:
        str: Table

    Raises:
        ValueError: if requested index results in a NaN in the computed features.
    """
    pipeline_features_row = pipeline_features.iloc[[index_to_explain]]
    input_features_row = input_features.iloc[[index_to_explain]]

    if algorithm == "shap":
        explainer_values, expected_value = _compute_shap_values(
            pipeline,
            pipeline_features_row,
            training_data=pipeline_features.dropna(axis=0),
        )
    elif algorithm == "lime":
        explainer_values = _compute_lime_values(
            pipeline,
            pipeline_features,
            index_to_explain,
        )
        expected_value = None
    else:
        raise ValueError(
            f"Unknown algorithm {algorithm}, should be one of ['shap', 'lime']",
        )
    normalized_values = _normalize_explainer_values(explainer_values)

    provenance = pipeline._get_feature_provenance()
    aggregated_explainer_values = _aggregate_explainer_values(
        explainer_values,
        provenance,
    )
    aggregated_normalized_explainer_values = _normalize_explainer_values(
        aggregated_explainer_values,
    )

    class_names = None
    if hasattr(pipeline, "classes_"):
        class_names = pipeline.classes_

    table_makers = {
        ProblemTypes.REGRESSION: _RegressionExplanationTable(
            top_k,
            include_explainer_values,
            include_expected_value,
            provenance,
            algorithm,
        ),
        ProblemTypes.BINARY: _BinaryExplanationTable(
            top_k,
            include_explainer_values,
            include_expected_value,
            class_names,
            provenance,
            algorithm,
        ),
        ProblemTypes.MULTICLASS: _MultiClassExplanationTable(
            top_k,
            include_explainer_values,
            include_expected_value,
            class_names,
            provenance,
            algorithm,
        ),
        ProblemTypes.TIME_SERIES_REGRESSION: _RegressionExplanationTable(
            top_k,
            include_explainer_values,
            include_expected_value,
            provenance,
            algorithm,
        ),
        ProblemTypes.TIME_SERIES_BINARY: _BinaryExplanationTable(
            top_k,
            include_explainer_values,
            include_expected_value,
            class_names,
            provenance,
            algorithm,
        ),
        ProblemTypes.TIME_SERIES_MULTICLASS: _MultiClassExplanationTable(
            top_k,
            include_explainer_values,
            include_expected_value,
            class_names,
            provenance,
            algorithm,
        ),
    }

    table_maker_class = table_makers[pipeline.problem_type]
    table_maker = {
        "text": table_maker_class.make_text,
        "dict": table_maker_class.make_dict,
        "dataframe": table_maker_class.make_dataframe,
    }[output_format]

    return table_maker(
        aggregated_explainer_values,
        aggregated_normalized_explainer_values,
        explainer_values,
        normalized_values,
        pipeline_features_row,
        input_features_row,
        expected_value,
    )


class _SectionMaker(abc.ABC):
    """Makes a section for a prediction explanations report.

    A report is made up of three parts: the header, the predicted values (if any), and the table.

    Each subclass of this class will be responsible for creating one of these sections.
    """

    @abc.abstractmethod
    def make_text(self, *args, **kwargs):
        """Makes the report section formatted as text."""

    @abc.abstractmethod
    def make_dict(self, *args, **kwargs):
        """Makes the report section formatted as a dictionary."""

    @abc.abstractmethod
    def make_dataframe(self, *args, **kwargs):
        """Makes the report section formatted as a dataframe."""


class _Heading(_SectionMaker):
    def __init__(self, prefixes, n_indices):
        self.prefixes = prefixes
        self.n_indices = n_indices

    def make_text(self, rank):
        """Makes the heading section for reports formatted as text.

        Differences between best/worst reports and reports where user manually specifies the input features subset
        are handled by formatting the value of the prefix parameter in the initialization.

        Args:
            rank (int): Rank (1, 2, 3, ...) of the prediction. Used to say "Best 1 of 5", "Worst 1 of 5", etc.

        Returns:
            The heading section for reports formatted as text.
        """
        prefix = self.prefixes[(rank // self.n_indices)]
        rank = rank % self.n_indices
        return [f"\t{prefix}{rank + 1} of {self.n_indices}\n\n"]

    def make_dict(self, rank):
        """Makes the heading section for reports formatted as a dictionary.

        Args:
            rank (int): Rank (1, 2, 3, ...) of the prediction. Used to say "Best 1 of 5", "Worst 1 of 5", etc.

        Returns:
            The heading section for reports formatted as a dictionary.
        """
        prefix = self.prefixes[(rank // self.n_indices)]
        rank = rank % self.n_indices
        return {"prefix": prefix, "index": rank + 1}

    def make_dataframe(self, rank):
        """Makes the heading section for reports formatted as a dataframe.

        Args:
            rank (int): Rank (1, 2, 3, ...) of the prediction. Used to say "Best 1 of 5", "Worst 1 of 5", etc.

        Returns:
            The heading section for reports formatted as a dictionary.
        """
        return self.make_dict(rank)


class _ClassificationPredictedValues(_SectionMaker):
    """Makes the predicted values section for classification problem best/worst reports formatted as text."""

    def __init__(self, error_name, y_pred_values):
        # Replace the default name with something more user-friendly
        if error_name == "cross_entropy":
            error_name = "Cross Entropy"
        self.error_name = error_name
        self.predicted_values = y_pred_values

    def make_text(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for classification problem best/worst reports formatted as text.

        Args:
            index (int): The index of the prediction in the dataset.
            y_pred (pd.Series): Pipeline predictions on the entire dataset.
            y_true (pd.Series): Targets for the entire dataset.
            scores (np.ndarray): Scores on the entire dataset.
            dataframe_index (pd.Series): pandas index for the entire dataset. Used to display the index in the data
                each explanation belongs to.

        Returns:
            The predicted values section for classification problem best/worst reports formatted as text.
        """
        pred_value = [
            f"{col_name}: {pred}"
            for col_name, pred in zip(
                y_pred.columns,
                round(y_pred.iloc[index], 3).tolist(),
            )
        ]
        pred_value = "[" + ", ".join(pred_value) + "]"
        true_value = y_true.iloc[index]

        return [
            f"\t\tPredicted Probabilities: {pred_value}\n",
            f"\t\tPredicted Value: {self.predicted_values.iloc[index]}\n",
            f"\t\tTarget Value: {true_value}\n",
            f"\t\t{self.error_name}: {round(scores[index], 3)}\n",
            f"\t\tIndex ID: {dataframe_index.iloc[index]}\n\n",
        ]

    def make_dict(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for classification problem best/worst reports formatted as dictionary."""
        pred_values = dict(zip(y_pred.columns, round(y_pred.iloc[index], 3).tolist()))

        return {
            "probabilities": pred_values,
            "predicted_value": _make_json_serializable(
                self.predicted_values.iloc[index],
            ),
            "target_value": _make_json_serializable(y_true.iloc[index]),
            "error_name": self.error_name,
            "error_value": _make_json_serializable(scores[index]),
            "index_id": _make_json_serializable(dataframe_index.iloc[index]),
        }

    def make_dataframe(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for classification problem best/worst reports formatted as dataframe."""
        return self.make_dict(index, y_pred, y_true, scores, dataframe_index)


class _RegressionPredictedValues(_SectionMaker):
    def __init__(self, error_name, y_pred_values=None):
        # Replace the default name with something more user-friendly
        if error_name == "abs_error":
            error_name = "Absolute Difference"
        self.error_name = error_name

    def make_text(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for regression problem best/worst reports formatted as text.

        Args:
            index (int): The index of the prediction in the dataset.
            y_pred (pd.Series): Pipeline predictions on the entire dataset.
            y_true (pd.Series): Targets for the entire dataset.
            scores (pd.Series): Scores on the entire dataset.
            dataframe_index (pd.Series): pandas index for the entire dataset. Used to display the index in the data
                each explanation belongs to.

        Returns:
            The predicted values section for regression problem best/worst reports formatted as text.
        """
        return [
            f"\t\tPredicted Value: {round(y_pred.iloc[index], 3)}\n",
            f"\t\tTarget Value: {round(y_true.iloc[index], 3)}\n",
            f"\t\t{self.error_name}: {round(scores[index], 3)}\n",
            f"\t\tIndex ID: {dataframe_index.iloc[index]}\n\n",
        ]

    def make_dict(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for regression problem best/worst reports formatted as a dictionary."""
        return {
            "probabilities": None,
            "predicted_value": round(y_pred.iloc[index], 3),
            "target_value": round(y_true.iloc[index], 3),
            "error_name": self.error_name,
            "error_value": round(scores[index], 3),
            "index_id": _make_json_serializable(dataframe_index.iloc[index]),
        }

    def make_dataframe(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section formatted as a dataframe."""
        dict_output = self.make_dict(index, y_pred, y_true, scores, dataframe_index)
        dict_output.pop("probabilities")
        return dict_output


class _ExplanationTable(_SectionMaker):
    def __init__(self, top_k_features, include_explainer_values, algorithm="shap"):
        self.top_k_features = top_k_features
        self.include_explainer_values = include_explainer_values
        self.algorithm = algorithm

    def make_text(self, index, pipeline, pipeline_features, input_features):
        """Makes the explanation table section for reports formatted as text.

        The table is the same whether the user requests a best/worst report or they manually specified the
        subset of the input features.

        Handling the differences in how the table is formatted between regression and classification problems
        is delegated to the _make_single_prediction_explanation_table

        Args:
            index (int): The index of the prediction in the dataset.
            pipeline (PipelineBase): The pipeline to explain.
            pipeline_features (pd.DataFrame): The dataframe of features created by the pipeline.
            input_features (pd.Dataframe): The dataframe of features passed to the pipeline.

        Returns:
            The explanation table section for reports formatted as text.
        """
        table = _make_single_prediction_explanation_table(
            pipeline,
            pipeline_features,
            input_features,
            index_to_explain=index,
            top_k=self.top_k_features,
            include_explainer_values=self.include_explainer_values,
            output_format="text",
            algorithm=self.algorithm,
        )
        table = table.splitlines()
        # Indent the rows of the table to match the indentation of the entire report.
        return ["\t\t" + line + "\n" for line in table] + ["\n\n"]

    def make_dict(self, index, pipeline, pipeline_features, input_features):
        """Makes the explanation table section formatted as a dictionary."""
        json_output = _make_single_prediction_explanation_table(
            pipeline,
            pipeline_features,
            input_features,
            index_to_explain=index,
            top_k=self.top_k_features,
            include_explainer_values=self.include_explainer_values,
            output_format="dict",
            algorithm=self.algorithm,
        )
        return json_output

    def make_dataframe(self, index, pipeline, pipeline_features, input_features):
        """Makes the explanation table section formatted as a dataframe."""
        return _make_single_prediction_explanation_table(
            pipeline,
            pipeline_features,
            input_features,
            index_to_explain=index,
            top_k=self.top_k_features,
            include_explainer_values=self.include_explainer_values,
            output_format="dataframe",
            algorithm=self.algorithm,
        )


class _ReportMaker:
    """Make a prediction explanation report.

    A report is made up of three parts: the header, the predicted values (if any), and the table.

    There are two kinds of reports we make: Reports where we explain the best and worst predictions and
    reports where we explain predictions for features the user has manually selected.

    Each of these reports is slightly different depending on the type of problem (regression, binary, multiclass).

    Rather than addressing all cases in one function/class, we write individual classes for formatting each part
    of the report depending on the type of problem and report.

    This class creates the report given callables for creating the header, predicted values, and table.
    """

    def __init__(self, heading_maker, predicted_values_maker, table_maker):
        self.heading_maker = heading_maker
        self.make_predicted_values_maker = predicted_values_maker
        self.table_maker = table_maker

    def make_text(self, data):
        """Make a prediction explanation report that is formatted as text.

        Args:
            data (_ReportData): Data passed in by the user.

        Returns:
             str
        """
        report = [data.pipeline.name + "\n\n", str(data.pipeline.parameters) + "\n\n"]
        for rank, index in enumerate(data.index_list):
            report.extend(self.heading_maker.make_text(rank))
            if self.make_predicted_values_maker:
                report.extend(
                    self.make_predicted_values_maker.make_text(
                        index,
                        data.y_pred,
                        data.y_true,
                        data.errors,
                        pd.Series(data.pipeline_features.index),
                    ),
                )
            else:
                report.extend([""])
            report.extend(
                self.table_maker.make_text(
                    index,
                    data.pipeline,
                    data.pipeline_features,
                    data.input_features,
                ),
            )
        return "".join(report)

    def make_dict(self, data):
        """Make a prediction explanation report that is formatted as a dictionary.

        Args:
            data (_ReportData): Data passed in by the user.

        Returns:
             dict
        """
        report = []
        for rank, index in enumerate(data.index_list):
            section = {}
            # We want to omit heading and predicted values sections for "explain_predictions"-style reports
            if self.heading_maker:
                section["rank"] = self.heading_maker.make_dict(rank)
            if self.make_predicted_values_maker:
                section[
                    "predicted_values"
                ] = self.make_predicted_values_maker.make_dict(
                    index,
                    data.y_pred,
                    data.y_true,
                    data.errors,
                    pd.Series(data.pipeline_features.index),
                )
            section["explanations"] = self.table_maker.make_dict(
                index,
                data.pipeline,
                data.pipeline_features,
                data.input_features,
            )["explanations"]
            report.append(section)
        return {"explanations": report}

    def make_dataframe(self, data):
        report = []
        for rank, index in enumerate(data.index_list):
            explanation_table = self.table_maker.make_dataframe(
                index,
                data.pipeline,
                data.pipeline_features,
                data.input_features,
            )
            if self.make_predicted_values_maker:
                heading = self.make_predicted_values_maker.make_dataframe(
                    index,
                    data.y_pred,
                    data.y_true,
                    data.errors,
                    pd.Series(data.pipeline_features.index),
                )
                for key, value in heading.items():
                    if key == "probabilities":
                        for class_name, probability in value.items():
                            explanation_table[
                                f"label_{class_name}_probability"
                            ] = probability
                    else:
                        explanation_table[key] = value
            if self.heading_maker:
                heading = self.heading_maker.make_dataframe(rank)
                explanation_table["rank"] = heading["index"]
                explanation_table["prefix"] = heading["prefix"]
            else:
                explanation_table["prediction_number"] = rank

            report.append(explanation_table)
        df = pd.concat(report).reset_index(drop=True)
        return df
