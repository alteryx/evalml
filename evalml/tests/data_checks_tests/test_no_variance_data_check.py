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

cases = [(all_distinct_X, all_distinct_y, True, []),
         ([1, 2, 3, 4], [1, 2, 3, 2], False, []),
         (np.arange(12).reshape(4, 3), [1, 2, 3], True, []),
         (all_null_X, all_distinct_y, False, [DataCheckError("Column feature has 0 unique value.", "NoVarianceDataCheck")]),
         (all_null_X, [1] * 4, False, [DataCheckError("Column feature has 0 unique value.", "NoVarianceDataCheck"),
                                       DataCheckError("The Labels have 1 unique value.", "NoVarianceDataCheck")]),
         (all_null_X, all_distinct_y, True, [DataCheckError("Column feature has 1 unique value.", "NoVarianceDataCheck")]),
         (all_distinct_X, all_null_y, True, [DataCheckError("The Labels have 1 unique value.", "NoVarianceDataCheck")]),
         (all_distinct_X, all_null_y, False, [DataCheckError("The Labels have 0 unique value.", "NoVarianceDataCheck")]),
         (two_distinct_with_nulls_X, two_distinct_with_nulls_y, True,
          [DataCheckWarning("Column feature has two unique values including nulls. Consider encoding the nulls for "
                            "this column to be useful for machine learning.", "NoVarianceDataCheck"),
           DataCheckWarning("The Labels have two unique values including nulls. Consider encoding the nulls for "
                            "this column to be useful for machine learning.", "NoVarianceDataCheck")
           ]),
         (two_distinct_with_nulls_X, two_distinct_with_nulls_y, False,
          [DataCheckError("Column feature has 1 unique value.", "NoVarianceDataCheck"),
           DataCheckError("The Labels have 1 unique value.", "NoVarianceDataCheck")])
         ]


@pytest.mark.parametrize("X,y,countna,answer", cases)
def test_no_variance_data_check_warnings(X, y, countna, answer):

    check = NoVarianceDataCheck(countna)
    assert check.validate(X, y) == answer
