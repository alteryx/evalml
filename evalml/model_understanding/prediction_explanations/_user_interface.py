import abc

import pandas as pd
from texttable import Texttable

from evalml.model_understanding.prediction_explanations._algorithms import (
    _compute_shap_values,
    _normalize_shap_values
)
from evalml.problem_types import ProblemTypes


def _make_rows(shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False,
               convert_numeric_to_string=True):
    """Makes the rows (one row for each feature) for the SHAP table.

    Arguments:
        shap_values (dict): Dictionary mapping the feature names to their SHAP values. In a multiclass setting,
            this dictionary for correspond to the SHAP values for a single class.
        normalized_values (dict): Normalized SHAP values. Same structure as shap_values parameter.
        top_k (int): How many of the highest/lowest features to include in the table.
        include_shap_values (bool): Whether to include the SHAP values in their own column.
        convert_numeric_to_string (bool): Whether numeric values should be converted to strings from numeric

    Returns:
          list(str)
    """
    tuples = [(value[0], feature_name) for feature_name, value in normalized_values.items()]

    # Sort the features s.t the top_k w the largest shap value magnitudes are the first
    # top_k elements
    tuples = sorted(tuples, key=lambda x: abs(x[0]), reverse=True)

    # Then sort such that the SHAP values go from most positive to most negative
    features_to_display = reversed(sorted(tuples[:top_k]))

    rows = []
    for value, feature_name in features_to_display:
        symbol = "+" if value >= 0 else "-"
        display_text = symbol * min(int(abs(value) // 0.2) + 1, 5)
        feature_value = pipeline_features[feature_name].iloc[0]
        if convert_numeric_to_string:
            if pd.api.types.is_number(feature_value) and not pd.api.types.is_bool(feature_value):
                feature_value = "{:.2f}".format(feature_value)
            else:
                feature_value = str(feature_value)
        row = [feature_name, feature_value, display_text]
        if include_shap_values:
            shap_value = shap_values[feature_name][0]
            if convert_numeric_to_string:
                shap_value = "{:.2f}".format(shap_value)
            row.append(shap_value)
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

    return {"feature_names": feature_names, "feature_values": feature_values,
            "qualitative_explanation": qualitative_explanations,
            "quantitative_explanation": quantitative_explanations}


def _make_json_serializable(value):
    """Make sure a numeric boolean type is json serializable.

    numpy.int64 or numpy.bool can't be serialized to json.
    """
    if pd.api.types.is_number(value):
        if pd.api.types.is_integer(value):
            value = int(value)
        else:
            value = float(value)
    elif pd.api.types.is_bool(value):
        value = bool(value)
    return value


def _make_text_table(shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
    """Make a table displaying the SHAP values for a prediction.

    Arguments:
        shap_values (dict): Dictionary mapping the feature names to their SHAP values. In a multiclass setting,
            this dictionary for correspond to the SHAP values for a single class.
        normalized_values (dict): Normalized SHAP values. Same structure as shap_values parameter.
        top_k (int): How many of the highest/lowest features to include in the table.
        include_shap_values (bool): Whether to include the SHAP values in their own column.

    Returns:
        str
    """
    n_cols = 4 if include_shap_values else 3
    dtypes = ["t"] * n_cols
    alignment = ["c"] * n_cols

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(dtypes)
    table.set_cols_align(alignment)

    header = ["Feature Name", "Feature Value", "Contribution to Prediction"]
    if include_shap_values:
        header.append("SHAP Value")

    rows = [header]
    rows += _make_rows(shap_values, normalized_values, pipeline_features, top_k, include_shap_values)
    table.add_rows(rows)
    return table.draw()


class _TableMaker(abc.ABC):
    """Makes a SHAP table for a regression, binary, or multiclass classification problem."""

    @abc.abstractmethod
    def make_text(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        """Creates a table given shap values and formats it as text."""

    @abc.abstractmethod
    def make_dict(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        """Creates a table given shap values and formats it as dictionary."""


class _RegressionSHAPTable(_TableMaker):
    """Makes a SHAP table explaining a prediction for a regression problems."""

    def make_text(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        return _make_text_table(shap_values, normalized_values, pipeline_features, top_k, include_shap_values)

    def make_dict(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        rows = _make_rows(shap_values, normalized_values, pipeline_features, top_k, include_shap_values,
                          convert_numeric_to_string=False)
        json_rows = _rows_to_dict(rows)
        json_rows["class_name"] = None
        return {"explanations": [json_rows]}


class _BinarySHAPTable(_TableMaker):
    """Makes a SHAP table explaining a prediction for a binary classification problem."""

    def __init__(self, class_names):
        self.class_names = class_names

    def make_text(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        # The SHAP algorithm will return a two-element list for binary problems.
        # By convention, we display the explanation for the dominant class.
        return _make_text_table(shap_values[1], normalized_values[1], pipeline_features, top_k, include_shap_values)

    def make_dict(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        rows = _make_rows(shap_values[1], normalized_values[1], pipeline_features, top_k, include_shap_values,
                          convert_numeric_to_string=False)
        json_rows = _rows_to_dict(rows)
        json_rows["class_name"] = _make_json_serializable(self.class_names[1])
        return {"explanations": [json_rows]}


class _MultiClassSHAPTable(_TableMaker):
    """Makes a SHAP table explaining a prediction for a multiclass classification problem."""

    def __init__(self, class_names):
        self.class_names = class_names

    def make_text(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        strings = []
        for class_name, class_values, normalized_class_values in zip(self.class_names, shap_values, normalized_values):
            strings.append(f"Class: {class_name}\n")
            table = _make_text_table(class_values, normalized_class_values, pipeline_features, top_k, include_shap_values)
            strings += table.splitlines()
            strings.append("\n")
        return "\n".join(strings)

    def make_dict(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        json_output = []
        for class_name, class_values, normalized_class_values in zip(self.class_names, shap_values, normalized_values):
            rows = _make_rows(class_values, normalized_class_values, pipeline_features, top_k, include_shap_values,
                              convert_numeric_to_string=False)
            json_output_for_class = _rows_to_dict(rows)
            json_output_for_class["class_name"] = _make_json_serializable(class_name)
            json_output.append(json_output_for_class)
        return {"explanations": json_output}


def _make_single_prediction_shap_table(pipeline, input_features, top_k=3, training_data=None,
                                       include_shap_values=False, output_format="text"):
    """Creates table summarizing the top_k positive and top_k negative contributing features to the prediction of a single datapoint.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of features - needs to correspond to data the pipeline was fit on.
        top_k (int): How many of the highest/lowest features to include in the table.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
            This is required for non-tree estimators because we need a sample of training data for the KernelSHAP algorithm.
        include_shap_values (bool): Whether the SHAP values should be included in an extra column in the output.
            Default is False.

    Returns:
        str: Table
    """
    pipeline_features = pipeline.compute_estimator_features(input_features)

    shap_values = _compute_shap_values(pipeline, pipeline_features, training_data)
    normalized_shap_values = _normalize_shap_values(shap_values)

    class_names = None
    if hasattr(pipeline, "classes_"):
        class_names = pipeline.classes_

    table_makers = {ProblemTypes.REGRESSION: _RegressionSHAPTable(),
                    ProblemTypes.BINARY: _BinarySHAPTable(class_names),
                    ProblemTypes.MULTICLASS: _MultiClassSHAPTable(class_names)}

    table_maker_class = table_makers[pipeline.problem_type]

    table_maker = table_maker_class.make_text if output_format == "text" else table_maker_class.make_dict

    return table_maker(shap_values, normalized_shap_values, pipeline_features, top_k, include_shap_values)


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


class _Heading(_SectionMaker):

    def __init__(self, prefixes, n_indices):
        self.prefixes = prefixes
        self.n_indices = n_indices

    def make_text(self, rank):
        """Makes the heading section for reports formatted as text.

        Differences between best/worst reports and reports where user manually specifies the input features subset
        are handled by formatting the value of the prefix parameter in the initialization.
        """
        prefix = self.prefixes[(rank // self.n_indices)]
        rank = rank % self.n_indices
        return [f"\t{prefix}{rank + 1} of {self.n_indices}\n\n"]

    def make_dict(self, rank):
        """Makes the heading section for reports formatted as dictionaries."""
        prefix = self.prefixes[(rank // self.n_indices)]
        rank = rank % self.n_indices
        return {"prefix": prefix, "index": rank + 1}


class _ClassificationPredictedValues(_SectionMaker):
    """Makes the predicted values section for classification problem best/worst reports formatted as text."""

    def __init__(self, error_name, y_pred_values):
        # Replace the default name with something more user-friendly
        if error_name == "cross_entropy":
            error_name = "Cross Entropy"
        self.error_name = error_name
        self.predicted_values = y_pred_values

    def make_text(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for classification problem best/worst reports formatted as text."""
        pred_value = [f"{col_name}: {pred}" for col_name, pred in
                      zip(y_pred.columns, round(y_pred.iloc[index], 3).tolist())]
        pred_value = "[" + ", ".join(pred_value) + "]"
        true_value = y_true.iloc[index]

        return [f"\t\tPredicted Probabilities: {pred_value}\n",
                f"\t\tPredicted Value: {self.predicted_values[index]}\n",
                f"\t\tTarget Value: {true_value}\n",
                f"\t\t{self.error_name}: {round(scores.iloc[index], 3)}\n",
                f"\t\tIndex ID: {dataframe_index.iloc[index]}\n\n"]

    def make_dict(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for classification problem best/worst reports formatted as dictionary."""
        pred_values = dict(zip(y_pred.columns, round(y_pred.iloc[index], 3).tolist()))

        return {"probabilities": pred_values,
                "predicted_value": _make_json_serializable(self.predicted_values[index]),
                "target_value": _make_json_serializable(y_true.iloc[index]),
                "error_name": self.error_name,
                "error_value": _make_json_serializable(scores.iloc[index]),
                "index_id": _make_json_serializable(dataframe_index.iloc[index])}


class _RegressionPredictedValues(_SectionMaker):
    def __init__(self, error_name, y_pred_values=None):
        # Replace the default name with something more user-friendly
        if error_name == "abs_error":
            error_name = "Absolute Difference"
        self.error_name = error_name

    def make_text(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for regression problem best/worst reports formatted as text."""
        return [f"\t\tPredicted Value: {round(y_pred.iloc[index], 3)}\n",
                f"\t\tTarget Value: {round(y_true.iloc[index], 3)}\n",
                f"\t\t{self.error_name}: {round(scores.iloc[index], 3)}\n",
                f"\t\tIndex ID: {dataframe_index.iloc[index]}\n\n"]

    def make_dict(self, index, y_pred, y_true, scores, dataframe_index):
        """Makes the predicted values section for regression problem best/worst reports formatted as a dictionary."""
        return {"probabilities": None, "predicted_value": round(y_pred.iloc[index], 3),
                "target_value": round(y_true.iloc[index], 3), "error_name": self.error_name,
                "error_value": round(scores.iloc[index], 3),
                "index_id": _make_json_serializable(dataframe_index.iloc[index])}


class _SHAPTable(_SectionMaker):
    def __init__(self, top_k_features, include_shap_values, training_data):
        self.top_k_features = top_k_features
        self.include_shap_values = include_shap_values
        self.training_data = training_data

    def make_text(self, index, pipeline, input_features):
        """Makes the SHAP table section for reports formatted as text.

        The table is the same whether the user requests a best/worst report or they manually specified the
        subset of the input features.

        Handling the differences in how the table is formatted between regression and classification problems
        is delegated to the _make_single_prediction_shap_table
        """
        table = _make_single_prediction_shap_table(pipeline, input_features.iloc[index:(index + 1)],
                                                   training_data=self.training_data, top_k=self.top_k_features,
                                                   include_shap_values=self.include_shap_values, output_format="text")
        table = table.splitlines()
        # Indent the rows of the table to match the indentation of the entire report.
        return ["\t\t" + line + "\n" for line in table] + ["\n\n"]

    def make_dict(self, index, pipeline, input_features):
        """Makes the SHAP table section formatted as a dictionary."""
        json_output = _make_single_prediction_shap_table(pipeline, input_features.iloc[index:(index + 1)],
                                                         training_data=self.training_data, top_k=self.top_k_features,
                                                         include_shap_values=self.include_shap_values,
                                                         output_format="dict")
        return json_output


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

        Arguments:
           data (_ReportData): Data passed in by the user.

        Returns:
             str
        """
        report = [data.pipeline.name + "\n\n", str(data.pipeline.parameters) + "\n\n"]
        for rank, index in enumerate(data.index_list):
            report.extend(self.heading_maker.make_text(rank))
            if self.make_predicted_values_maker:
                report.extend(self.make_predicted_values_maker.make_text(index, data.y_pred, data.y_true,
                                                                         data.errors,
                                                                         pd.Series(data.input_features.index)))
            else:
                report.extend([""])
            report.extend(self.table_maker.make_text(index, data.pipeline, data.input_features))
        return "".join(report)

    def make_dict(self, data):
        """Make a prediction explanation report that is formatted as a dictionary.

        Arguments:
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
                section["predicted_values"] = self.make_predicted_values_maker.make_dict(index, data.y_pred,
                                                                                         data.y_true, data.errors,
                                                                                         pd.Series(data.input_features.index))
            section["explanations"] = self.table_maker.make_dict(index, data.pipeline,
                                                                 data.input_features)["explanations"]
            report.append(section)
        return {"explanations": report}
