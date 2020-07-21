from texttable import Texttable


def _make_rows(normalized_values, top_k):
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
        rows.append([feature_name, display_text])

    return rows


def _make_single_prediction_table(normalized_values, top_k=3):
    """Makes a table from the normalized prediction explanation values for a single prediction.

    Arguments:
        normalized_values (dict): Dictionary mapping a feature name to a one-element list
            containing its scaled value.
        top_k (int): Will include the top_k highest and lowest features in the table.

    Returns:
        str
    """

    rows = [["Feature Name", "Contribution to Prediction"]]

    rows += _make_rows(normalized_values, top_k)

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t', 't'])
    table.set_cols_align(["c", "c"])
    table.add_rows(rows)
    return table.draw()
