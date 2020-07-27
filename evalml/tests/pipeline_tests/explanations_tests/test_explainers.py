from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.prediction_explanations._explainers import (
    _explain_prediction
)

test_features = [5, [1], np.ones((1, 15)), pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).iloc[0],
                 pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), pd.DataFrame()]


@pytest.mark.parametrize("test_features", test_features)
@patch("evalml.pipelines.prediction_explanations._explainers._compute_shap_values")
@patch("evalml.pipelines.prediction_explanations._explainers._normalize_shap_values")
def test_explain_prediction_value_error(mock_normalize_shap_values, mock_compute_shap_values, test_features):
    with pytest.raises(ValueError, match="features must be stored in a dataframe of one row."):
        _explain_prediction(None, input_features=test_features, training_data=None)


@patch("evalml.pipelines.prediction_explanations._explainers._compute_shap_values", return_value={"a": [1], "b": [-2],
                                                                                                  "c": [-0.25], "d": [2]})
@patch("evalml.pipelines.prediction_explanations._explainers._normalize_shap_values", return_value={"a": [0.5],
                                                                                                    "b": [-0.75],
                                                                                                    "c": [-0.25],
                                                                                                    "d": [0.75]})
def test_explain_prediction_runs(mock_normalize_shap_values, mock_compute_shap_values):

    answer = """Feature Name   Contribution to Prediction
        =========================================
         d                    ++++
         a                    +++
         c                     --
         b                    ----""".splitlines()

    pipeline = MagicMock()
    features = pd.DataFrame({"a": [1], "b": [2]})
    table = _explain_prediction(pipeline, features).splitlines()

    assert len(table) == len(answer)
    for row, row_answer in zip(table, answer):
        assert row.strip().split() == row_answer.strip().split()
