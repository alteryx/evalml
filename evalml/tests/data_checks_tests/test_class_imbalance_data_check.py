import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import (
    ClassImbalanceDataCheck,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)

class_imbalance_data_check_name = ClassImbalanceDataCheck.name


def test_class_imbalance_errors():
    X = pd.DataFrame()

    with pytest.raises(ValueError, match="threshold 0 is not within the range"):
        ClassImbalanceDataCheck(threshold=0).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="threshold 0.51 is not within the range"):
        ClassImbalanceDataCheck(threshold=0.51).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="threshold -0.5 is not within the range"):
        ClassImbalanceDataCheck(threshold=-0.5).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="Provided number of CV folds"):
        ClassImbalanceDataCheck(num_cv_folds=-1).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="Provided value min_samples"):
        ClassImbalanceDataCheck(min_samples=-1).validate(X, y=pd.Series([0, 1, 1]))


@pytest.mark.parametrize("test_size", [-1, "a", 0, 12, 3.14])
def test_class_imbalance_data_check_validates_test_size(test_size):
    with pytest.raises(
        ValueError,
        match="Parameter test_size must be a number between 0 and less than or equal to 1",
    ):
        ClassImbalanceDataCheck(test_size=test_size)


@pytest.mark.parametrize("input_type", ["pd", "np", "ww"])
@pytest.mark.parametrize("test_size", [1, 0.5, 0.2])
def test_class_imbalance_data_check_binary(test_size, input_type):
    X = pd.DataFrame()
    y = pd.Series([0, 0, 1] * int(1 / test_size))
    y_long = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * int(1 / test_size))
    y_balanced = pd.Series([0, 0, 1, 1] * int(1 / test_size))

    if input_type == "np":
        X = X.to_numpy()
        y = y.to_numpy()
        y_long = y_long.to_numpy()
        y_balanced = y_balanced.to_numpy()

    elif input_type == "ww":
        X.ww.init()
        y = ww.init_series(y, logical_type="integer")
        y_long = ww.init_series(y_long, logical_type="integer")
        y_balanced = ww.init_series(y_balanced, logical_type="integer")

    class_imbalance_check = ClassImbalanceDataCheck(
        min_samples=1,
        num_cv_folds=0,
        test_size=test_size,
    )
    assert class_imbalance_check.validate(X, y) == []
    assert class_imbalance_check.validate(X, y_long) == [
        DataCheckWarning(
            message="The following labels fall below 10% of the target: [0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [0]},
        ).to_dict(),
    ]
    assert ClassImbalanceDataCheck(
        threshold=0.25,
        min_samples=1,
        num_cv_folds=0,
        test_size=test_size,
    ).validate(X, y_long) == [
        DataCheckWarning(
            message="The following labels fall below 25% of the target: [0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [0]},
        ).to_dict(),
    ]

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1, test_size=test_size)
    assert class_imbalance_check.validate(X, y) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [1]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [1]},
        ).to_dict(),
    ]

    assert class_imbalance_check.validate(X, y_balanced) == []

    class_imbalance_check = ClassImbalanceDataCheck(test_size=test_size)
    assert class_imbalance_check.validate(X, y) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0, 1]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0, 1]},
        ).to_dict(),
    ]


@pytest.mark.parametrize("input_type", ["pd", "np", "ww"])
@pytest.mark.parametrize("test_size", [1, 0.5, 0.2])
def test_class_imbalance_data_check_multiclass(test_size, input_type):
    X = pd.DataFrame()
    y = pd.Series([0, 2, 1, 1] * int(1 / test_size))
    y_imbalanced_default_threshold = pd.Series(
        [0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * int(1 / test_size),
    )
    y_imbalanced_set_threshold = pd.Series(
        [0, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * int(1 / test_size),
    )
    y_imbalanced_cv = pd.Series([0, 1, 2, 2, 1, 1, 1] * int(1 / test_size))
    y_long = pd.Series(
        [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4] * int(1 / test_size),
    )

    if input_type == "np":
        X = X.to_numpy()
        y = y.to_numpy()
        y_imbalanced_default_threshold = y_imbalanced_default_threshold.to_numpy()
        y_imbalanced_set_threshold = y_imbalanced_set_threshold.to_numpy()
        y_imbalanced_cv = y_imbalanced_cv.to_numpy()
        y_long = y_long.to_numpy()

    elif input_type == "ww":
        X.ww.init()
        y = ww.init_series(y, logical_type="integer")
        y_imbalanced_default_threshold = ww.init_series(
            y_imbalanced_default_threshold,
            logical_type="integer",
        )
        y_imbalanced_set_threshold = ww.init_series(
            y_imbalanced_set_threshold,
            logical_type="integer",
        )
        y_imbalanced_cv = ww.init_series(y_imbalanced_cv, logical_type="integer")
        y_long = ww.init_series(y_long, logical_type="integer")

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0, test_size=test_size)
    assert class_imbalance_check.validate(X, y) == []
    assert class_imbalance_check.validate(X, y_imbalanced_default_threshold) == [
        DataCheckWarning(
            message="The following labels fall below 10% of the target: [0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [0]},
        ).to_dict(),
        DataCheckWarning(
            message="The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
            details={"target_values": [0]},
        ).to_dict(),
    ]

    assert ClassImbalanceDataCheck(
        threshold=0.25,
        num_cv_folds=0,
        min_samples=1,
        test_size=test_size,
    ).validate(X, y_imbalanced_set_threshold) == [
        DataCheckWarning(
            message="The following labels fall below 25% of the target: [3, 0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [3, 0]},
        ).to_dict(),
    ]

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=2, test_size=test_size)
    assert class_imbalance_check.validate(X, y_imbalanced_cv) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [0, 2]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0, 2]},
        ).to_dict(),
    ]

    assert class_imbalance_check.validate(X, y_long) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [0, 1]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0, 1]},
        ).to_dict(),
    ]

    class_imbalance_check = ClassImbalanceDataCheck(test_size=test_size)
    assert class_imbalance_check.validate(X, y_long) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0, 1, 2, 3]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0, 1, 2, 3]},
        ).to_dict(),
    ]


@pytest.mark.parametrize("input_type", ["pd", "np", "ww"])
@pytest.mark.parametrize("test_size", [1, 0.5, 0.2])
def test_class_imbalance_empty_and_nan(test_size, input_type):
    X = pd.DataFrame()
    y_empty = pd.Series([])
    y_has_nan = pd.Series(
        [np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2] * int(1 / test_size),
    )

    if input_type == "np":
        X = X.to_numpy()
        y_empty = y_empty.to_numpy()
        y_has_nan = y_has_nan.to_numpy()

    elif input_type == "ww":
        X.ww.init()
        y_empty = ww.init_series(y_empty)
        y_has_nan = ww.init_series(y_has_nan)
    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0, test_size=test_size)

    assert class_imbalance_check.validate(X, y_empty) == []
    res = ClassImbalanceDataCheck(
        threshold=0.5,
        min_samples=1,
        num_cv_folds=0,
        test_size=test_size,
    ).validate(X, y_has_nan)

    expected = [
        DataCheckWarning(
            message="The following labels fall below 50% of the target: [2]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [2]},
        ).to_dict(),
    ]
    assert res == expected

    res = ClassImbalanceDataCheck(
        threshold=0.5,
        num_cv_folds=0,
        test_size=test_size,
    ).validate(X, y_has_nan)

    expected = [
        DataCheckWarning(
            message="The following labels fall below 50% of the target: [2]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [2]},
        ).to_dict(),
        DataCheckWarning(
            message="The following labels in the target have severe class imbalance because they fall under 50% of the target and have less than 100 samples: [2]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
            details={"target_values": [2]},
        ).to_dict(),
    ]

    assert res == expected

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1, test_size=test_size)
    assert class_imbalance_check.validate(X, y_empty) == []

    res = ClassImbalanceDataCheck(
        threshold=0.5,
        num_cv_folds=1,
        test_size=test_size,
    ).validate(X, y_has_nan)

    expected = [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [2]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [2]},
        ).to_dict(),
        DataCheckWarning(
            message="The following labels fall below 50% of the target: [2]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [2]},
        ).to_dict(),
        DataCheckWarning(
            message="The following labels in the target have severe class imbalance because they fall under 50% of the target and have less than 100 samples: [2]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
            details={"target_values": [2]},
        ).to_dict(),
    ]
    assert res == expected


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("test_size", [1, 0.5, 0.2])
def test_class_imbalance_nonnumeric(test_size, input_type):
    X = pd.DataFrame()
    y_bools = pd.Series([True, False, False, False, False] * int(1 / test_size))
    y_binary = pd.Series(["yes", "no", "yes", "yes", "yes"] * int(1 / test_size))
    y_multiclass = pd.Series(
        [
            "red",
            "green",
            "red",
            "red",
            "blue",
            "green",
            "red",
            "blue",
            "green",
            "red",
            "red",
            "red",
        ]
        * int(1 / test_size),
    )
    y_multiclass_imbalanced_folds = pd.Series(
        ["No", "Maybe", "Maybe", "No", "Yes"] * int(1 / test_size),
    )
    y_binary_imbalanced_folds = pd.Series(
        ["No", "Yes", "No", "Yes", "No"] * int(1 / test_size),
    )

    if input_type == "ww":
        X.ww.init()
        y_bools = ww.init_series(y_bools, logical_type="boolean")
        y_binary = ww.init_series(y_binary, logical_type="categorical")
        y_multiclass = ww.init_series(y_multiclass, logical_type="categorical")

    class_imbalance_check = ClassImbalanceDataCheck(
        threshold=0.25,
        min_samples=1,
        num_cv_folds=0,
        test_size=test_size,
    )
    assert class_imbalance_check.validate(X, y_bools) == [
        DataCheckWarning(
            message="The following labels fall below 25% of the target: [True]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [True]},
        ).to_dict(),
    ]

    assert class_imbalance_check.validate(X, y_binary) == [
        DataCheckWarning(
            message="The following labels fall below 25% of the target: ['no']",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": ["no"]},
        ).to_dict(),
    ]

    assert ClassImbalanceDataCheck(
        threshold=0.35,
        num_cv_folds=0,
        test_size=test_size,
    ).validate(X, y_multiclass) == [
        DataCheckWarning(
            message="The following labels fall below 35% of the target: ['green', 'blue']",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": ["green", "blue"]},
        ).to_dict(),
        DataCheckWarning(
            message="The following labels in the target have severe class imbalance because they fall under 35% of the target and have less than 100 samples: ['green', 'blue']",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
            details={"target_values": ["green", "blue"]},
        ).to_dict(),
    ]

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1, test_size=test_size)
    assert class_imbalance_check.validate(X, y_multiclass_imbalanced_folds) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: ['Yes']",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": ["Yes"]},
        ).to_dict(),
    ]
    assert class_imbalance_check.validate(X, y_multiclass) == []

    class_imbalance_check = ClassImbalanceDataCheck(test_size=test_size)
    assert class_imbalance_check.validate(X, y_binary_imbalanced_folds) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['No', 'Yes']",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": ["No", "Yes"]},
        ).to_dict(),
    ]

    assert class_imbalance_check.validate(X, y_multiclass) == [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['blue', 'green']",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": ["blue", "green"]},
        ).to_dict(),
    ]


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("test_size", [1, 0.5, 0.2])
def test_class_imbalance_nonnumeric_balanced(test_size, input_type):
    X = pd.DataFrame()
    y_bools_balanced = pd.Series([True, True, True, False, False] * int(1 / test_size))
    y_binary_balanced = pd.Series(["No", "Yes", "No", "Yes"] * int(1 / test_size))
    y_multiclass_balanced = pd.Series(
        ["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"]
        * int(1 / test_size),
    )
    if input_type == "ww":
        X.ww.init()
        y_bools_balanced = ww.init_series(y_bools_balanced, logical_type="boolean")
        y_binary_balanced = ww.init_series(
            y_binary_balanced,
            logical_type="categorical",
        )
        y_multiclass_balanced = ww.init_series(y_multiclass_balanced, "categorical")

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1, test_size=test_size)
    assert class_imbalance_check.validate(X, y_multiclass_balanced) == []
    assert class_imbalance_check.validate(X, y_binary_balanced) == []
    assert class_imbalance_check.validate(X, y_multiclass_balanced) == []
    assert class_imbalance_check.validate(X, y_bools_balanced) == []


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("min_samples", [1, 20, 50, 100, 500])
@pytest.mark.parametrize("test_size", [1, 0.5, 0.2])
def test_class_imbalance_severe(test_size, min_samples, input_type):
    X = pd.DataFrame()
    # 0 will be < 10% of the data, but there will be 50 samples of it
    y_values_binary = pd.Series(
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 50 * int(1 / test_size),
    )
    y_values_multiclass = pd.Series(
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        * 50
        * int(1 / test_size),
    )
    if input_type == "ww":
        X.ww.init()
        y_values_binary = ww.init_series(y_values_binary, logical_type="integer")
        y_values_multiclass = ww.init_series(
            y_values_multiclass,
            logical_type="integer",
        )

    class_imbalance_check = ClassImbalanceDataCheck(
        min_samples=min_samples,
        num_cv_folds=1,
        test_size=test_size,
    )
    warnings = [
        DataCheckWarning(
            message="The following labels fall below 10% of the target: [0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [0]},
        ).to_dict(),
    ]
    if min_samples > 50:
        warnings.append(
            DataCheckWarning(
                message=f"The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than {min_samples} samples: [0]",
                data_check_name=class_imbalance_data_check_name,
                message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                details={"target_values": [0]},
            ).to_dict(),
        )
    assert class_imbalance_check.validate(X, y_values_binary) == warnings
    assert class_imbalance_check.validate(X, y_values_multiclass) == warnings


@pytest.mark.parametrize("test_size", [1, 0.5, 0.2])
def test_class_imbalance_large_multiclass(test_size):
    X = pd.DataFrame()
    y_values_multiclass_large = pd.Series(
        [0] * 20 + [1] * 25 + [2] * 99 + [3] * 105 + [4] * 900 + [5] * 900,
    )
    y_multiclass_huge = pd.Series([i % 200 for i in range(100000)])
    y_imbalanced_multiclass_huge = y_multiclass_huge.append(
        pd.Series([200] * 10),
        ignore_index=True,
    )
    y_imbalanced_multiclass_nan = y_multiclass_huge.append(
        pd.Series([np.nan] * 10),
        ignore_index=True,
    )

    y_values_multiclass_large = y_values_multiclass_large.tolist() * int(1 / test_size)
    y_multiclass_huge = y_multiclass_huge.tolist() * int(1 / test_size)
    y_imbalanced_multiclass_huge = y_imbalanced_multiclass_huge.tolist() * int(
        1 / test_size,
    )
    y_imbalanced_multiclass_nan = y_imbalanced_multiclass_nan.tolist() * int(
        1 / test_size,
    )

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1, test_size=test_size)
    assert class_imbalance_check.validate(X, y_values_multiclass_large) == [
        DataCheckWarning(
            message="The following labels fall below 10% of the target: [2, 1, 0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [2, 1, 0]},
        ).to_dict(),
        DataCheckWarning(
            message=f"The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [2, 1, 0]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
            details={"target_values": [2, 1, 0]},
        ).to_dict(),
    ]

    assert class_imbalance_check.validate(X, y_multiclass_huge) == []

    assert class_imbalance_check.validate(X, y_imbalanced_multiclass_huge) == [
        DataCheckWarning(
            message="The following labels fall below 10% of the target: [200]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
            details={"target_values": [200]},
        ).to_dict(),
        DataCheckWarning(
            message=f"The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [200]",
            data_check_name=class_imbalance_data_check_name,
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
            details={"target_values": [200]},
        ).to_dict(),
    ]

    assert class_imbalance_check.validate(X, y_imbalanced_multiclass_nan) == []
