import numpy as np
import pandas as pd
import pytest

from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
)
from evalml.data_checks.no_variance_data_check import NoVarianceDataCheck

all_distinct_X = pd.DataFrame({"feature": [1, 2, 3, 4]})
all_null_X = pd.DataFrame({"feature": [None] * 4,
                           "feature_2": list(range(4))})
two_distinct_with_nulls_X = pd.DataFrame({"feature": [1, 1, None, None],
                                          "feature_2": list(range(4))})

all_distinct_y = pd.Series([1, 2, 3, 4])
all_null_y = pd.Series([None] * 4)
two_distinct_with_nulls_y = pd.Series(([1] * 2) + ([None] * 2))

feature_0_unique = DataCheckError("Column feature has 0 unique value.", "NoVarianceDataCheck")
feature_1_unique = DataCheckError("Column feature has 1 unique value.", "NoVarianceDataCheck")
labels_0_unique = DataCheckError("The Labels have 0 unique value.", "NoVarianceDataCheck")
labels_1_unique = DataCheckError("The Labels have 1 unique value.", "NoVarianceDataCheck")

cases = [(all_distinct_X, all_distinct_y, True, []),
         ([1, 2, 3, 4], [1, 2, 3, 2], False, []),
         (np.arange(12).reshape(4, 3), [1, 2, 3], True, []),
         (all_null_X, all_distinct_y, False, [feature_0_unique]),
         (all_null_X, [1] * 4, False, [feature_0_unique, labels_1_unique]),
         (all_null_X, all_distinct_y, True, [feature_1_unique]),
         (all_distinct_X, all_null_y, True, [labels_1_unique]),
         (all_distinct_X, all_null_y, False, [labels_0_unique]),
         (two_distinct_with_nulls_X, two_distinct_with_nulls_y, True,
          [DataCheckWarning("Column feature has two unique values including nulls. Consider encoding the nulls for "
                            "this column to be useful for machine learning.", "NoVarianceDataCheck"),
           DataCheckWarning("The Labels have two unique values including nulls. Consider encoding the nulls for "
                            "this column to be useful for machine learning.", "NoVarianceDataCheck")
           ]),
         (two_distinct_with_nulls_X, two_distinct_with_nulls_y, False, [feature_1_unique, labels_1_unique])
         ]


@pytest.mark.parametrize("X,y,countna,answer", cases)
def test_no_variance_data_check_warnings(X, y, countna, answer):
    check = NoVarianceDataCheck(countna)
    assert check.validate(X, y) == answer
