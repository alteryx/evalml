import pandas as pd
import pytest
import woodwork

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    MulticollinearityDataCheck,
)

multi_data_check_name = MulticollinearityDataCheck.name


def test_multicollinearity_data_check_init():
    multi_check = MulticollinearityDataCheck()
    assert multi_check.threshold == 0.9

    multi_check = MulticollinearityDataCheck(threshold=0.0)
    assert multi_check.threshold == 0

    multi_check = MulticollinearityDataCheck(threshold=0.5)
    assert multi_check.threshold == 0.5

    multi_check = MulticollinearityDataCheck(threshold=1.0)
    assert multi_check.threshold == 1.0

    with pytest.raises(
        ValueError,
        match="threshold must be a float between 0 and 1, inclusive.",
    ):
        MulticollinearityDataCheck(threshold=-0.1)
    with pytest.raises(
        ValueError,
        match="threshold must be a float between 0 and 1, inclusive.",
    ):
        MulticollinearityDataCheck(threshold=1.1)


@pytest.mark.parametrize("use_nullable_types", [True, False])
def test_multicollinearity_returns_warning(use_nullable_types):
    data = pd.Series(range(30))
    col = pd.Series(data)
    X = pd.DataFrame(
        {
            "col_1": col,
            "col_2": col * 3,
            "col_3": ~col,
            "col_4": col / 2,
            "col_5": col + 1,
            "not_collinear": [0, 1, 0, 0, 0] * 6,
        },
    )
    if use_nullable_types:
        data[1] = None
        X["col_nullable"] = woodwork.init_series(
            pd.Series(data),
            logical_type="IntegerNullable",
        )
        X.ww.init()

    collinear_cols = [
        c for c in X.columns if c not in {"not_collinear", "col_nullable"}
    ]
    msg_list = []
    for idx, col_1 in enumerate(collinear_cols):
        for col_2 in collinear_cols[idx + 1 : :]:
            if col_1 != col_2:
                msg_list.append((col_1, col_2))

    message = str(msg_list)

    multi_check = MulticollinearityDataCheck(threshold=0.95)
    assert multi_check.validate(X) == [
        DataCheckWarning(
            message="Columns are likely to be correlated: %s" % message,
            data_check_name=multi_data_check_name,
            message_code=DataCheckMessageCode.IS_MULTICOLLINEAR,
            details={"columns": msg_list},
        ).to_dict(),
    ]


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_multicollinearity_nonnumeric_cols(data_type, make_data_type):
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "a"],
            "col_2": ["w", "x", "y", "z", "b"],
            "col_3": ["a", "a", "c", "d", "a"],
            "col_4": ["a", "b", "c", "d", "a"],
            "col_5": ["0", "0", "1", "2", "0"],
            "col_6": [1, 1, 2, 3, 1],
        },
    )
    X = pd.concat([X] * 10, axis=0).reset_index(drop=True)
    X.ww.init(
        logical_types={
            "col_1": "categorical",
            "col_2": "categorical",
            "col_3": "categorical",
            "col_4": "categorical",
            "col_5": "categorical",
        },
    )

    multi_check = MulticollinearityDataCheck(threshold=0.9)
    assert multi_check.validate(X) == [
        DataCheckWarning(
            message="Columns are likely to be correlated: [('col_1', 'col_4'), ('col_3', 'col_5'), ('col_3', 'col_6'), ('col_5', 'col_6')]",
            data_check_name=multi_data_check_name,
            message_code=DataCheckMessageCode.IS_MULTICOLLINEAR,
            details={
                "columns": [
                    ("col_1", "col_4"),
                    ("col_3", "col_5"),
                    ("col_3", "col_6"),
                    ("col_5", "col_6"),
                ],
            },
        ).to_dict(),
    ]


def test_multicollinearity_data_check_input_formats():
    multi_check = MulticollinearityDataCheck(threshold=0.9)

    # test empty pd.DataFrame
    assert multi_check.validate(pd.DataFrame()) == []
