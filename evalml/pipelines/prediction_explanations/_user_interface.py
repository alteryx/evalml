from texttable import Texttable


def _make_rows(shap_values, normalized_values, top_k, include_shap_values=False):
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
        row = [feature_name, display_text]
        if include_shap_values:
            row.append(round(shap_values[feature_name][0], 2))
        rows.append(row)

    return rows


def _make_table(shap_values, normalized_values, top_k, include_shap_values=False):
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
    dtypes = ["t", "t"]
    alignment = ["c", "c"]

    if include_shap_values:
        dtypes.append("f")
        alignment.append("c")

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(dtypes)
    table.set_cols_align(alignment)

    header = ["Feature Name", "Contribution to Prediction"]
    if include_shap_values:
        header.append("SHAP Value")

    rows = [header]
    rows += _make_rows(shap_values, normalized_values, top_k, include_shap_values)
    table.add_rows(rows)
    return table.draw()


class _SHAPRegressionTableMaker:
    """Makes a SHAP table explaining a prediction for a regression problems."""

    def __call__(self, shap_values, normalized_values, top_k, include_shap_values=False):
        return _make_table(shap_values, normalized_values, top_k, include_shap_values)


class _SHAPBinaryTableMaker:
    """Makes a SHAP table explaining a prediction for a binary classification problem."""

    def __call__(self, shap_values, normalized_values, top_k, include_shap_values=False):
        # The SHAP algorithm will return a two-element list for binary problems.
        # By convention, we display the explanation for the dominant class.
        return _make_table(shap_values[1], normalized_values[1], top_k, include_shap_values)


class _SHAPMultiClassTableMaker:
    """Makes a SHAP table explaining a prediction for a multiclass classification problem."""

    def __init__(self, class_names):
        self.class_names = class_names

    def __call__(self, shap_values, normalized_values, top_k, include_shap_values=False):
        strings = []
        for class_name, class_values, normalized_class_values in zip(self.class_names, shap_values, normalized_values):
            strings.append(f"Class: {class_name}\n")
            table = _make_table(class_values, normalized_class_values, top_k, include_shap_values)
            strings += table.splitlines()
            strings.append("\n")
        return "\n".join(strings)


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
