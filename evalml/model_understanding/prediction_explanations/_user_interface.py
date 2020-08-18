import abc

import numpy as np
import pandas as pd
from texttable import Texttable

from evalml.model_understanding.prediction_explanations._algorithms import (
    _compute_shap_values,
    _normalize_shap_values
)
from evalml.problem_types import ProblemTypes


def _make_rows(shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
    """Makes the rows (one row for each feature) for the SHAP table.

    Arguments:
        shap_values (dict): Dictionary mapping the feature names to their SHAP values. In a multiclass setting,
            this dictionary for correspond to the SHAP values for a single class.
        normalized_values (dict): Normalized SHAP values. Same structure as shap_values parameter.
        top_k (int): How many of the highest/lowest features to include in the table.
        include_shap_values (bool): Whether to include the SHAP values in their own column.

    Returns:
          list(str)
    """
    tuples = [(value[0], feature_name) for feature_name, value in normalized_values.items()]
    tuples = sorted(tuples)

    if len(tuples) <= 2 * top_k:
        features_to_display = reversed(tuples)
    else:
        features_to_display = tuples[-top_k:][::-1] + tuples[:top_k][::-1]

    rows = []
    for value, feature_name in features_to_display:
        symbol = "+" if value >= 0 else "-"
        display_text = symbol * min(int(abs(value) // 0.2) + 1, 5)
        feature_value = pipeline_features[feature_name]
        if pd.api.types.is_number(feature_value):
            feature_value = np.round(feature_value, 2)
        else:
            feature_value = str(feature_value)
        row = [feature_name, feature_value, display_text]
        if include_shap_values:
            row.append(np.round(shap_values[feature_name][0], 2))
        rows.append(row)

    return rows


def _jsonify_rows(rows):
    """Turns a list of lists into a json-friendly dictionary."""

    feature_names = []
    feature_values = []
    qualitative_explanations = []
    quantitative_explanations = []
    for name, value, qualitative, quantitative in rows:
        feature_names.append(name)
        feature_values.append(value)
        qualitative_explanations.append(qualitative)
        quantitative_explanations.append(quantitative)

    return {"feature_names": feature_names, "feature_values": feature_values,
            "qualitative_explanation": qualitative_explanations,
            "quantitative_explanation": quantitative_explanations}


def _make_table(shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
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
    dtypes = ["t", "f", "t", "f"] if include_shap_values else ["t", "t", "t"]
    alignment = ["c", "c", "c", "c"] if include_shap_values else ["c", "c", "c"]

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
    def __call__(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        """Creates a table given shap values."""


class _SHAPRegressionTableMaker(_TableMaker):
    """Makes a SHAP table explaining a prediction for a regression problems."""

    def __call__(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        return _make_table(shap_values, normalized_values, pipeline_features, top_k, include_shap_values)


class _SHAPBinaryTableMaker(_TableMaker):
    """Makes a SHAP table explaining a prediction for a binary classification problem."""

    def __call__(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        # The SHAP algorithm will return a two-element list for binary problems.
        # By convention, we display the explanation for the dominant class.
        return _make_table(shap_values[1], normalized_values[1], pipeline_features, top_k, include_shap_values)


class _SHAPMultiClassTableMaker(_TableMaker):
    """Makes a SHAP table explaining a prediction for a multiclass classification problem."""

    def __init__(self, class_names):
        self.class_names = class_names

    def __call__(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        strings = []
        for class_name, class_values, normalized_class_values in zip(self.class_names, shap_values, normalized_values):
            strings.append(f"Class: {class_name}\n")
            table = _make_table(class_values, normalized_class_values, pipeline_features, top_k, include_shap_values)
            strings += table.splitlines()
            strings.append("\n")
        return "\n".join(strings)


class _SHAPRegressionJSONMaker(_TableMaker):

    def __call__(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        rows = _make_rows(shap_values, normalized_values, pipeline_features, top_k, include_shap_values)
        json_rows = _jsonify_rows(rows)
        json_rows["class_name"] = None
        return {"explanation": [json_rows]}


class _SHAPBinaryJSONMaker(_TableMaker):

    def __call__(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        rows = _make_rows(shap_values[1], normalized_values[1], pipeline_features, top_k, include_shap_values)
        json_rows = _jsonify_rows(rows)
        json_rows["class_name"] = None
        return {"explanation": [json_rows]}


class _SHAPMultiClassJSONMaker(_TableMaker):

    def __init__(self, class_names):
        self.class_names = class_names

    def __call__(self, shap_values, normalized_values, pipeline_features, top_k, include_shap_values=False):
        json_output = []
        for class_name, class_values, normalized_class_values in zip(self.class_names, shap_values, normalized_values):
            rows = _make_rows(class_values, normalized_class_values, pipeline_features, top_k, include_shap_values)
            json_output_for_class = _jsonify_rows(rows)
            json_output_for_class["class_name"] = class_name
            json_output.append(json_output_for_class)
        return {"explanation": json_output}


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
    if not (isinstance(input_features, pd.DataFrame) and input_features.shape[0] == 1):
        raise ValueError("features must be stored in a dataframe of one row.")

    shap_values = _compute_shap_values(pipeline, input_features, training_data)
    normalized_shap_values = _normalize_shap_values(shap_values)

    # We need a dict of type {column_name: feature value}
    pipeline_features = pipeline._transform(input_features)
    features_dict = dict(zip(pipeline_features.columns, *pipeline_features.values))

    table_makers = {("text", ProblemTypes.REGRESSION): _SHAPRegressionTableMaker(),
                    ("text", ProblemTypes.BINARY): _SHAPBinaryTableMaker(),
                    ("text", ProblemTypes.MULTICLASS): _SHAPMultiClassTableMaker(pipeline._classes),
                    ("json", ProblemTypes.REGRESSION): _SHAPRegressionJSONMaker(),
                    ("json", ProblemTypes.BINARY): _SHAPBinaryJSONMaker(),
                    ("json", ProblemTypes.MULTICLASS): _SHAPMultiClassJSONMaker(pipeline._classes)}

    table_maker = table_makers[(output_format, pipeline.problem_type)]

    return table_maker(shap_values, normalized_shap_values, features_dict, top_k, include_shap_values)


class _ReportSectionMaker:
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

    def make_report_section(self, pipeline, input_features, indices, y_pred, y_true, errors):
        """Make a report for a subset of input features to a fitted pipeline.

        Arguments:
            pipeline (PipelineBase): Fitted pipeline.
            input_features (pd.DataFrame): Features where the pipeline predictions will be explained.
            indices (list(int)): List of indices specifying the subset of input features whose predictions
                we want to explain.
            y_pred (pd.Series): Predicted values of the input_features.
            y_true (pd.Series): True labels of the input_features.
            errors (pd.Series): Error between y_pred and y_true

        Returns:
             str
        """
        report = []
        for rank, index in enumerate(indices):
            report.extend(self.heading_maker(rank, index))
            report.extend(self.make_predicted_values_maker(index, y_pred, y_true, errors))
            report.extend(self.table_maker(index, pipeline, input_features))
        return report


class _JSONReportMaker:
    def __init__(self, heading_maker, predicted_values_maker, table_maker):
        self.heading_maker = heading_maker
        self.make_predicted_values_maker = predicted_values_maker
        self.table_maker = table_maker

    def make_report_section(self, pipeline, input_features, indices, y_pred, y_true, errors):
        """Make a report for a subset of input features to a fitted pipeline.

        Arguments:
            pipeline (PipelineBase): Fitted pipeline.
            input_features (pd.DataFrame): Features where the pipeline predictions will be explained.
            indices (list(int)): List of indices specifying the subset of input features whose predictions
                we want to explain.
            y_pred (pd.Series): Predicted values of the input_features.
            y_true (pd.Series): True labels of the input_features.
            errors (pd.Series): Error between y_pred and y_true

        Returns:
             str
        """
        report = []
        for rank, index in enumerate(indices):
            section = {}
            section["rank"] = self.heading_maker(rank, index)
            section["predicted_values"] = self.make_predicted_values_maker(index, y_pred, y_true, errors)
            section["explanation"] = self.table_maker(index, pipeline, input_features)["explanation"]
            report.append(section)
        return {"explanations": report}


class _SectionMaker(abc.ABC):
    """Makes a section for a prediction explanations report.

    A report is made up of three parts: the header, the predicted values (if any), and the table.

    Each subclass of this class will be responsible for creating one of these sections.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Makes the report section.

        Returns:
            list(str): A list containing the lines of report section.
        """


class _HeadingMaker(_SectionMaker):
    """Makes the heading section for reports.

    Differences between best/worst reports and reports where user manually specifies the input features subset
    are handled by formatting the value of the prefix parameter in the initialization.
    """

    def __init__(self, prefix, n_indices):
        self.prefix = prefix
        self.n_indices = n_indices

    def __call__(self, rank, index):
        return [f"\t{self.prefix}{rank + 1} of {self.n_indices}\n\n"]


class _JSONHeadingMaker(_SectionMaker):
    def __init__(self, prefix, n_indices):
        self.prefix = prefix
        self.n_indices = n_indices

    def __call__(self, rank, index):
        return {"prefix": self.prefix, "index": rank + 1}


class _EmptyPredictedValuesMaker(_SectionMaker):
    """Omits the predicted values section for reports where the user specifies the subset of the input features."""

    def __call__(self, index, y_pred, y_true, scores):
        return [""]


class _EmptyJSONPredictedValuesMaker(_SectionMaker):
    def __call__(self, index, y_pred, y_true, scores):
        return {"probabilities": None, "predicted_value": None, "target_value": None,
                "error_name": None, "error_value": None}


class _ClassificationPredictedValuesMaker(_SectionMaker):
    """Makes the predicted values section for classification problem best/worst reports."""

    def __init__(self, error_name, y_pred_values):
        # Replace the default name with something more user-friendly
        if error_name == "cross_entropy":
            error_name = "Cross Entropy"
        self.error_name = error_name
        self.predicted_values = y_pred_values

    def __call__(self, index, y_pred, y_true, scores):
        pred_value = [f"{col_name}: {pred}" for col_name, pred in
                      zip(y_pred.columns, round(y_pred.iloc[index], 3).tolist())]
        pred_value = "[" + ", ".join(pred_value) + "]"
        true_value = y_true[index]

        return [f"\t\tPredicted Probabilities: {pred_value}\n",
                f"\t\tPredicted Value: {self.predicted_values[index]}\n",
                f"\t\tTarget Value: {true_value}\n",
                f"\t\t{self.error_name}: {round(scores[index], 3)}\n\n"]


class _ClassificationJSONPredictedValuesMaker(_SectionMaker):
    def __init__(self, error_name, y_pred_values):
        # Replace the default name with something more user-friendly
        if error_name == "cross_entropy":
            error_name = "Cross Entropy"
        self.error_name = error_name
        self.predicted_values = y_pred_values

    def __call__(self, index, y_pred, y_true, scores):
        pred_values = dict(zip(y_pred.columns, round(y_pred.iloc[index], 3).tolist()))

        return {"probabilities": pred_values, "predicted_value": self.predicted_values[index],
                "target_value": y_true[index], "error_name": self.error_name, "error_value": round(scores[index], 3)}


class _RegressionPredictedValuesMaker(_SectionMaker):
    """Makes the predicted values section for regression problem best/worst reports."""

    def __init__(self, error_name):
        # Replace the default name with something more user-friendly
        if error_name == "abs_error":
            error_name = "Absolute Difference"
        self.error_name = error_name

    def __call__(self, index, y_pred, y_true, scores):

        return [f"\t\tPredicted Value: {round(y_pred.iloc[index], 3)}\n",
                f"\t\tTarget Value: {round(y_true[index], 3)}\n",
                f"\t\t{self.error_name}: {round(scores[index], 3)}\n\n"]


class _RegressionJSONPredictedValuesMaker(_SectionMaker):
    def __init__(self, error_name):
        # Replace the default name with something more user-friendly
        if error_name == "abs_error":
            error_name = "Absolute Difference"
        self.error_name = error_name

    def __call__(self, index, y_pred, y_true, scores):

        return {"probabilities": None, "predicted_value": round(y_pred.iloc[index], 3),
                "target_value": round(y_true[index], 3), "error_name": self.error_name,
                "error_value": round(scores[index], 3)}


class _SHAPTableMaker(_SectionMaker):
    """Makes the SHAP table section for reports.

    The table is the same whether the user requests a best/worst report or they manually specified the
    subset of the input features.

    Handling the differences in how the table is formatted between regression and classification problems
    is delegated to the explain_prediction function.
    """

    def __init__(self, top_k_features, include_shap_values, training_data):
        self.top_k_features = top_k_features
        self.include_shap_values = include_shap_values
        self.training_data = training_data

    def __call__(self, index, pipeline, input_features):
        table = _make_single_prediction_shap_table(pipeline, input_features.iloc[index:(index + 1)],
                                                   training_data=self.training_data, top_k=self.top_k_features,
                                                   include_shap_values=self.include_shap_values)
        table = table.splitlines()
        # Indent the rows of the table to match the indentation of the entire report.
        return ["\t\t" + line + "\n" for line in table] + ["\n\n"]


class _JSONSHAPTableMaker(_SectionMaker):

    def __init__(self, top_k_features, include_shap_values, training_data):
        self.top_k_features = top_k_features
        self.include_shap_values = include_shap_values
        self.training_data = training_data

    def __call__(self, index, pipeline, input_features):
        json_output = _make_single_prediction_shap_table(pipeline, input_features.iloc[index:(index + 1)],
                                                         training_data=self.training_data, top_k=self.top_k_features,
                                                         include_shap_values=self.include_shap_values,
                                                         output_format="json")
        return json_output
