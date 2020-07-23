from texttable import Texttable


def _make_rows(shap_values, normalized_values, top_k, include_shap_values=False):
    tuples = [(value[0], feature_name) for feature_name, value in normalized_values.items()]
    tuples = sorted(tuples)

    if len(tuples) <= 2*top_k:
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


def _make_table(dtypes, alignment, shap_values, normalized_values, top_k, include_shap_values=False):
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


def _make_single_prediction_table(shap_values, normalized_values, top_k=3, include_shap_values=False):
    """Makes a table from the normalized prediction explanation values for a single prediction.

    Arguments:
        normalized_values (dict): Dictionary mapping a feature name to a one-element list
            containing its scaled value.
        top_k (int): Will include the top_k highest and lowest features in the table.

    Returns:
        str
    """
    dtypes = ["t", "t"]
    alignment = ["c", "c"]

    if include_shap_values:
        dtypes.append("f")
        alignment.append("c")

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            strings = ["Positive Label\n"]
            table = _make_table(dtypes, alignment, shap_values[1], normalized_values[1], top_k, include_shap_values)
            strings += table.splitlines()
            return "\n".join(strings)
        else:
            strings = []
            for class_index, (class_values, normalized_class_values) in enumerate(zip(shap_values, normalized_values)):
                strings.append(f"Class {class_index}\n")
                table = _make_table(dtypes, alignment, class_values, normalized_class_values, top_k, include_shap_values)
                strings += table.splitlines()
                strings.append("\n")
            return "\n".join(strings)
    else:
        return _make_table(dtypes, alignment, shap_values, normalized_values, top_k, include_shap_values)
