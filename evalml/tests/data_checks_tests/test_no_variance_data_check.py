import numpy as np
import pandas as pd
import pytest

from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
)
from evalml.data_checks.no_variance_data_check import NoVarianceDataCheck

NAME = NoVarianceDataCheck.name

all_distinct_X = pd.DataFrame({"feature": [1, 2, 3, 4]})
all_null_X = pd.DataFrame({"feature": [None] * 4,
                           "feature_2": list(range(4))})
two_distinct_with_nulls_X = pd.DataFrame({"feature": [1, 1, None, None],
                                          "feature_2": list(range(4))})

all_distinct_y = pd.Series([1, 2, 3, 4])
all_null_y = pd.Series([None] * 4)
two_distinct_with_nulls_y = pd.Series(([1] * 2) + ([None] * 2))
all_null_y_with_name = pd.Series([None] * 4)
all_null_y_with_name.name = "Labels"

feature_0_unique = DataCheckError("feature has 0 unique value.", NAME)
feature_1_unique = DataCheckError("feature has 1 unique value.", NAME)
labels_0_unique = DataCheckError("Y has 0 unique value.", NAME)
labels_1_unique = DataCheckError("Y has 1 unique value.", NAME)

cases = [(all_distinct_X, all_distinct_y, True, []),
         ([1, 2, 3, 4], [1, 2, 3, 2], False, []),
         (np.arange(12).reshape(4, 3), [1, 2, 3], True, []),
         (all_null_X, all_distinct_y, False, [feature_0_unique]),
         (all_null_X, [1] * 4, False, [feature_0_unique, labels_1_unique]),
         (all_null_X, all_distinct_y, True, [feature_1_unique]),
         (all_distinct_X, all_null_y, True, [labels_1_unique]),
         (all_distinct_X, all_null_y, False, [labels_0_unique]),
         (two_distinct_with_nulls_X, two_distinct_with_nulls_y, True,
          [DataCheckWarning("feature has two unique values including nulls. Consider encoding the nulls for "
                            "this column to be useful for machine learning.", NAME),
           DataCheckWarning("Y has two unique values including nulls. Consider encoding the nulls for "
                            "this column to be useful for machine learning.", NAME)
           ]),
         (two_distinct_with_nulls_X, two_distinct_with_nulls_y, False, [feature_1_unique, labels_1_unique]),
         (all_distinct_X, all_null_y_with_name, False, [DataCheckError("Labels has 0 unique value.", NAME)])
         ]


@pytest.mark.parametrize("X,y,countna,answer", cases)
def test_no_variance_data_check_warnings(X, y, countna, answer):
    check = NoVarianceDataCheck(countna)
    assert check.validate(X, y) == answer
