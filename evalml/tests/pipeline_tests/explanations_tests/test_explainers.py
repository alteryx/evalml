from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.explanations._explainers import _explain_prediction

test_cases = [5, [1], np.ones((1, 15)), pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).iloc[0]]


@pytest.mark.parametrize("test_case", test_cases)
def test_explain_prediction_value_error(test_case):
    with pytest.raises(ValueError, match="features must be stored in a dataframe of one row."):
        _explain_prediction(None, features=test_case, training_data=None)


@patch("evalml.pipelines.explanations._explainers._compute_shap_values", return_value={"a": [1], "b": [-2]})
def test_explain_prediction_runs(mock_compute_shap_values):
    pipeline = MagicMock()
    features = pd.DataFrame({"a": [1], "b": [2]})
    _explain_prediction(pipeline, features)
