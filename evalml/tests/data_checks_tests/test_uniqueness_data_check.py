import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    UniquenessDataCheck
)

uniqueness_data_check_name = UniquenessDataCheck.name


def test_uniqueness_data_check_init():
    uniqueness_check = UniquenessDataCheck("regression")
    assert uniqueness_check.threshold == 0.50

    uniqueness_check = UniquenessDataCheck("regression", threshold=0.0)
    assert uniqueness_check.threshold == 0

    uniqueness_check = UniquenessDataCheck("regression", threshold=0.5)
    assert uniqueness_check.threshold == 0.5

    uniqueness_check = UniquenessDataCheck("regression", threshold=1.0)
    assert uniqueness_check.threshold == 1.0

    with pytest.raises(ValueError, match="threshold must be a float between 0 and 1, inclusive."):
        UniquenessDataCheck("regression", threshold=-0.1)
    with pytest.raises(ValueError, match="threshold must be a float between 0 and 1, inclusive."):
        UniquenessDataCheck("regression", threshold=1.1)


def test_highly_null_data_check_warnings():
    data = pd.DataFrame({'regression_unique_enough': [float(x) for x in range(100)],
                         'regression_not_unique_enough': [float(1) for x in range(100)]})
    uniqueness_check = UniquenessDataCheck(problem_type="regression")
    assert uniqueness_check.validate(data) == {
        "warnings": [DataCheckWarning(message="Input columns (regression_not_unique_enough) for regression problem type are not unique enough.",
                                      data_check_name=uniqueness_data_check_name,
                                      message_code=DataCheckMessageCode.NOT_UNIQUE_ENOUGH,
                                      details={"column": "regression_not_unique_enough"}).to_dict()],
        "errors": []
    }

    data = pd.DataFrame({'multiclass_too_unique': ["Cats", "Are", "Absolutely", "The", "Best"] * 20,
                         'multiclass_not_too_unique': ["Cats", "Cats", "Best", "Best", "Best"] * 20})
    uniqueness_check = UniquenessDataCheck(problem_type="multiclass")
    assert uniqueness_check.validate(data) == {
        "warnings": [DataCheckWarning(
            message="Input columns (multiclass_too_unique) for multiclass problem type are too unique.",
            data_check_name=uniqueness_data_check_name,
            message_code=DataCheckMessageCode.TOO_UNIQUE,
            details={"column": "multiclass_too_unique"}).to_dict()],
        "errors": []
    }
