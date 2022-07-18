import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
    NullDataCheck,
)

highly_null_data_check_name = NullDataCheck.name


@pytest.fixture
def highly_null_dataframe():
    return pd.DataFrame(
        {
            "some_null": [2, 4, None, None, 5],
            "lots_of_null": [None, None, None, None, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        },
    )


@pytest.fixture
def highly_null_dataframe_nullable_types(highly_null_dataframe):
    df = highly_null_dataframe
    df.ww.init(
        logical_types={
            "lots_of_null": "IntegerNullable",
            "all_null": "IntegerNullable",
        },
    )
    return df


class SeriesWrap:
    def __init__(self, series):
        self.series = series

    def __eq__(self, series_2):
        return all(self.series.eq(series_2.series))


def test_highly_null_data_check_init():
    highly_null_check = NullDataCheck()
    assert highly_null_check.pct_null_col_threshold == 0.95
    assert highly_null_check.pct_moderately_null_col_threshold == 0.20
    assert highly_null_check.pct_null_row_threshold == 0.95

    highly_null_check = NullDataCheck(pct_null_col_threshold=0.40)
    assert highly_null_check.pct_null_col_threshold == 0.40
    assert highly_null_check.pct_moderately_null_col_threshold == 0.20
    assert highly_null_check.pct_null_row_threshold == 0.95

    highly_null_check = NullDataCheck(pct_moderately_null_col_threshold=0.50)
    assert highly_null_check.pct_null_col_threshold == 0.95
    assert highly_null_check.pct_moderately_null_col_threshold == 0.50
    assert highly_null_check.pct_null_row_threshold == 0.95

    highly_null_check = NullDataCheck(pct_null_row_threshold=0.5)
    assert highly_null_check.pct_null_col_threshold == 0.95
    assert highly_null_check.pct_moderately_null_col_threshold == 0.20
    assert highly_null_check.pct_null_row_threshold == 0.5

    highly_null_check = NullDataCheck(
        pct_null_col_threshold=1.0,
        pct_null_row_threshold=1.0,
    )
    assert highly_null_check.pct_null_col_threshold == 1.0
    assert highly_null_check.pct_null_row_threshold == 1.0

    with pytest.raises(
        ValueError,
        match="`pct_null_col_threshold` must be a float between 0 and 1, inclusive.",
    ):
        NullDataCheck(pct_null_col_threshold=-0.1)
    with pytest.raises(
        ValueError,
        match="`pct_null_col_threshold` must be a float between 0 and 1, inclusive.",
    ):
        NullDataCheck(pct_null_col_threshold=1.1)
    with pytest.raises(
        ValueError,
        match="`pct_moderately_null_col_threshold` must be a float between 0 and 1, inclusive, and must be less than or equal to `pct_null_col_threshold`.",
    ):
        NullDataCheck(pct_moderately_null_col_threshold=-0.1)
    with pytest.raises(
        ValueError,
        match="`pct_moderately_null_col_threshold` must be a float between 0 and 1, inclusive, and must be less than or equal to `pct_null_col_threshold`.",
    ):
        NullDataCheck(pct_moderately_null_col_threshold=1.1)
    with pytest.raises(
        ValueError,
        match="`pct_moderately_null_col_threshold` must be a float between 0 and 1, inclusive, and must be less than or equal to `pct_null_col_threshold`.",
    ):
        NullDataCheck(
            pct_null_col_threshold=0.90,
            pct_moderately_null_col_threshold=0.95,
        )
    with pytest.raises(
        ValueError,
        match="`pct_null_row_threshold` must be a float between 0 and 1, inclusive.",
    ):
        NullDataCheck(pct_null_row_threshold=-0.5)
    with pytest.raises(
        ValueError,
        match="`pct_null_row_threshold` must be a float between 0 and 1, inclusive.",
    ):
        NullDataCheck(pct_null_row_threshold=2.1)


@pytest.mark.parametrize("nullable_type", [True, False])
def test_highly_null_data_check_warnings(
    nullable_type,
    highly_null_dataframe_nullable_types,
    highly_null_dataframe,
):
    # Test the data check with nullable types being used.
    if nullable_type:
        df = highly_null_dataframe_nullable_types
    else:
        df = highly_null_dataframe
    no_null_check = NullDataCheck(
        pct_null_col_threshold=0.0,
        pct_moderately_null_col_threshold=0.0,
        pct_null_row_threshold=0.0,
    )
    highly_null_rows = SeriesWrap(pd.Series([2 / 4, 2 / 4, 3 / 4, 3 / 4, 1 / 4]))
    validate_messages = no_null_check.validate(df)
    validate_messages[0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_messages[0]["details"]["pct_null_cols"],
    )
    assert validate_messages == [
        DataCheckWarning(
            message="5 out of 5 rows are 0.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
            details={
                "pct_null_cols": highly_null_rows,
                "rows": highly_null_rows.series.index.tolist(),
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=highly_null_data_check_name,
                    metadata={"rows": [0, 1, 2, 3, 4]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'some_null', 'lots_of_null', 'all_null' are 0.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["some_null", "lots_of_null", "all_null"],
                "pct_null_rows": {
                    "some_null": 0.4,
                    "all_null": 1.0,
                    "lots_of_null": 0.8,
                },
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["some_null", "lots_of_null", "all_null"]},
                ),
            ],
        ).to_dict(),
    ]

    some_null_check = NullDataCheck(
        pct_null_col_threshold=0.5,
        pct_null_row_threshold=0.5,
    )
    highly_null_rows = SeriesWrap(pd.Series([2 / 4, 2 / 4, 3 / 4, 3 / 4]))
    validate_messages = some_null_check.validate(df)
    validate_messages[0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_messages[0]["details"]["pct_null_cols"],
    )
    assert validate_messages == [
        DataCheckWarning(
            message="4 out of 5 rows are 50.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
            details={"pct_null_cols": highly_null_rows, "rows": [0, 1, 2, 3]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=highly_null_data_check_name,
                    metadata={"rows": [0, 1, 2, 3]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'lots_of_null', 'all_null' are 50.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["lots_of_null", "all_null"],
                "pct_null_rows": {"all_null": 1.0, "lots_of_null": 0.8},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["lots_of_null", "all_null"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'some_null' have between 20.0% and 50.0% null values",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.COLS_WITH_NULL,
            details={
                "columns": ["some_null"],
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={
                        "columns": ["some_null"],
                        "is_target": False,
                        "rows": None,
                    },
                    parameters={
                        "impute_strategies": {
                            "parameter_type": "column",
                            "columns": {
                                "some_null": {
                                    "impute_strategy": {
                                        "categories": ["mean", "most_frequent"],
                                        "type": "category",
                                        "default_value": "mean",
                                    },
                                },
                            },
                        },
                    },
                ),
            ],
        ).to_dict(),
    ]

    all_null_check = NullDataCheck(
        pct_null_col_threshold=1.0,
        pct_null_row_threshold=1.0,
    )
    assert all_null_check.validate(df) == [
        DataCheckWarning(
            message="Column(s) 'all_null' are 100.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["all_null"],
                "pct_null_rows": {"all_null": 1.0},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["all_null"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'some_null', 'lots_of_null' have between 20.0% and 100.0% null values",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.COLS_WITH_NULL,
            details={
                "columns": ["some_null", "lots_of_null"],
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={
                        "columns": ["some_null", "lots_of_null"],
                        "is_target": False,
                    },
                    parameters={
                        "impute_strategies": {
                            "parameter_type": "column",
                            "columns": {
                                "lots_of_null": {
                                    "impute_strategy": {
                                        "categories": ["mean", "most_frequent"],
                                        "type": "category",
                                        "default_value": "mean",
                                    },
                                },
                                "some_null": {
                                    "impute_strategy": {
                                        "categories": ["mean", "most_frequent"],
                                        "type": "category",
                                        "default_value": "mean",
                                    },
                                },
                            },
                        },
                    },
                ),
            ],
        ).to_dict(),
    ]


def test_highly_null_data_check_separate_rows_cols(highly_null_dataframe):
    row_null_check = NullDataCheck(
        pct_null_col_threshold=0.9,
        pct_moderately_null_col_threshold=0.75,
        pct_null_row_threshold=0.0,
    )
    highly_null_rows = SeriesWrap(pd.Series([2 / 4, 2 / 4, 3 / 4, 3 / 4, 1 / 4]))
    validate_messages = row_null_check.validate(highly_null_dataframe)
    validate_messages[0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_messages[0]["details"]["pct_null_cols"],
    )
    assert validate_messages == [
        DataCheckWarning(
            message="5 out of 5 rows are 0.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
            details={"pct_null_cols": highly_null_rows, "rows": [0, 1, 2, 3, 4]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=highly_null_data_check_name,
                    metadata={"rows": [0, 1, 2, 3, 4]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'all_null' are 90.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["all_null"],
                "pct_null_rows": {"all_null": 1.0},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["all_null"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'lots_of_null' have between 75.0% and 90.0% null values",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.COLS_WITH_NULL,
            details={
                "columns": ["lots_of_null"],
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["lots_of_null"], "is_target": False},
                    parameters={
                        "impute_strategies": {
                            "parameter_type": "column",
                            "columns": {
                                "lots_of_null": {
                                    "impute_strategy": {
                                        "categories": ["mean", "most_frequent"],
                                        "type": "category",
                                        "default_value": "mean",
                                    },
                                },
                            },
                        },
                    },
                ),
            ],
        ).to_dict(),
    ]

    col_null_check = NullDataCheck(
        pct_null_col_threshold=0.0,
        pct_moderately_null_col_threshold=0.0,
        pct_null_row_threshold=0.9,
    )
    validate_messages = col_null_check.validate(highly_null_dataframe)
    assert validate_messages == [
        DataCheckWarning(
            message="Column(s) 'some_null', 'lots_of_null', 'all_null' are 0.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["some_null", "lots_of_null", "all_null"],
                "pct_null_rows": {
                    "some_null": 0.4,
                    "lots_of_null": 0.8,
                    "all_null": 1.0,
                },
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["some_null", "lots_of_null", "all_null"]},
                ),
            ],
        ).to_dict(),
    ]


def test_highly_null_data_check_input_formats():
    highly_null_check = NullDataCheck(
        pct_null_col_threshold=0.8,
        pct_null_row_threshold=0.8,
    )

    # test empty pd.DataFrame
    assert highly_null_check.validate(pd.DataFrame()) == []

    highly_null_rows = SeriesWrap(pd.Series([0.8]))
    expected = [
        DataCheckWarning(
            message="1 out of 2 rows are 80.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
            details={"pct_null_cols": highly_null_rows, "rows": [0]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=highly_null_data_check_name,
                    metadata={"rows": [0]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) '0', '1', '2' are 80.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": [0, 1, 2],
                "pct_null_rows": {0: 1.0, 1: 1.0, 2: 1.0},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": [0, 1, 2]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) '3' have between 20.0% and 80.0% null values",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.COLS_WITH_NULL,
            details={
                "columns": [3],
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": [3], "is_target": False},
                    parameters={
                        "impute_strategies": {
                            "parameter_type": "column",
                            "columns": {
                                3: {
                                    "impute_strategy": {
                                        "categories": ["most_frequent"],
                                        "type": "category",
                                        "default_value": "most_frequent",
                                    },
                                },
                            },
                        },
                    },
                ),
            ],
        ).to_dict(),
    ]
    #  test Woodwork
    ww_input = pd.DataFrame([[None, None, None, None, 0], [None, None, None, "hi", 5]])
    ww_input.ww.init(logical_types={1: "categorical", 3: "categorical"})
    validate_messages = highly_null_check.validate(ww_input)
    validate_messages[0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_messages[0]["details"]["pct_null_cols"],
    )
    assert validate_messages == expected


def test_get_null_column_information(highly_null_dataframe):
    (
        highly_null_cols,
        highly_null_cols_indices,
    ) = NullDataCheck.get_null_column_information(
        highly_null_dataframe,
        pct_null_col_threshold=0.8,
    )
    assert highly_null_cols == {"lots_of_null": 0.8, "all_null": 1.0}
    assert highly_null_cols_indices == {
        "lots_of_null": [0, 1, 2, 3],
        "all_null": [0, 1, 2, 3, 4],
    }


def test_get_null_row_information(highly_null_dataframe):
    expected_highly_null_rows = SeriesWrap(pd.Series([2 / 4, 2 / 4, 3 / 4, 3 / 4]))
    highly_null_rows = NullDataCheck.get_null_row_information(
        highly_null_dataframe,
        pct_null_row_threshold=0.5,
    )
    highly_null_rows = SeriesWrap(highly_null_rows)
    assert highly_null_rows == expected_highly_null_rows


def test_has_null_but_not_highly_null():
    X = pd.DataFrame(
        {
            "few_null_categorical": [None, "a", "b", "c", "d"],
            "few_null": [1, None, 3, 4, 5],
            "few_null_categorical_2": [None, "a", "b", "c", "d"],
            "few_null_2": [1, None, 3, 0, 5],
            "no_null": [1, 2, 3, 4, 5],
            "no_null_categorical": ["a", "b", "a", "d", "e"],
        },
    )
    X.ww.init(
        logical_types={
            "few_null_categorical": "categorical",
            "few_null_categorical_2": "categorical",
        },
    )

    null_check = NullDataCheck(pct_null_col_threshold=0.5, pct_null_row_threshold=1.0)
    validate_messages = null_check.validate(X)
    assert validate_messages == [
        DataCheckWarning(
            message="Column(s) 'few_null_categorical', 'few_null', 'few_null_categorical_2', 'few_null_2' have between 20.0% and 50.0% null values",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.COLS_WITH_NULL,
            details={
                "columns": [
                    "few_null_categorical",
                    "few_null",
                    "few_null_categorical_2",
                    "few_null_2",
                ],
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={
                        "columns": [
                            "few_null_categorical",
                            "few_null",
                            "few_null_categorical_2",
                            "few_null_2",
                        ],
                        "is_target": False,
                    },
                    parameters={
                        "impute_strategies": {
                            "parameter_type": "column",
                            "columns": {
                                "few_null_categorical": {
                                    "impute_strategy": {
                                        "categories": ["most_frequent"],
                                        "type": "category",
                                        "default_value": "most_frequent",
                                    },
                                },
                                "few_null": {
                                    "impute_strategy": {
                                        "categories": ["mean", "most_frequent"],
                                        "type": "category",
                                        "default_value": "mean",
                                    },
                                },
                                "few_null_categorical_2": {
                                    "impute_strategy": {
                                        "categories": ["most_frequent"],
                                        "type": "category",
                                        "default_value": "most_frequent",
                                    },
                                },
                                "few_null_2": {
                                    "impute_strategy": {
                                        "categories": ["mean", "most_frequent"],
                                        "type": "category",
                                        "default_value": "mean",
                                    },
                                },
                            },
                        },
                    },
                ),
            ],
        ).to_dict(),
    ]


def test_null_data_check_natural_language_highly_null_dropped():
    X = pd.DataFrame(
        {
            "few_null_natural_language": [None, "a", "b", "c", "d"],
            "highly_null_natural_language": [None, None, "b", None, "d"],
            "no_null_natural_language": ["a", "b", "a", "d", "e"],
        },
    )

    X.ww.init(
        logical_types={
            "highly_null_natural_language": "NaturalLanguage",
            "few_null_natural_language": "NaturalLanguage",
            "no_null_natural_language": "NaturalLanguage",
        },
    )

    null_check = NullDataCheck(pct_null_col_threshold=0.5, pct_null_row_threshold=1.0)
    validate_messages = null_check.validate(X)

    assert validate_messages == [
        DataCheckWarning(
            message="Column(s) 'highly_null_natural_language' are 50.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["highly_null_natural_language"],
                "pct_null_rows": {"highly_null_natural_language": 0.6},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["highly_null_natural_language"]},
                ),
            ],
        ).to_dict(),
    ]


def test_null_data_check_datetime_highly_null_dropped():
    X = pd.DataFrame()
    X["highly_null_datetime"] = pd.Series(pd.date_range("20200101", periods=5))
    for i in range(3):
        X.loc[i][0] = None

    X["few_null_datetime"] = pd.Series(pd.date_range("20200101", periods=5))
    X.loc[4][1] = None

    null_check = NullDataCheck(pct_null_col_threshold=0.5, pct_null_row_threshold=1.0)
    validate_messages = null_check.validate(X)

    assert validate_messages == [
        DataCheckWarning(
            message="Column(s) 'highly_null_datetime' are 50.0% or more null",
            data_check_name=highly_null_data_check_name,
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["highly_null_datetime"],
                "pct_null_rows": {"highly_null_datetime": 0.6},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=highly_null_data_check_name,
                    metadata={"columns": ["highly_null_datetime"]},
                ),
            ],
        ).to_dict(),
    ]
