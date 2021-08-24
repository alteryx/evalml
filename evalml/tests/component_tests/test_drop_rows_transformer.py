import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.pipelines.components.transformers.preprocessing import (
    DropRowsTransformer,
)


def test_drop_rows_transformer_init():
    X = pd.DataFrame({"a column": [1, 2, 3], "another col": [4, 5, 6]})
    drop_rows_transformer = DropRowsTransformer(indices_to_drop=[0, 1])
    assert drop_rows_transformer.indices_to_drop == [0, 1]


def test_drop_rows_transformer_fit_transform():
    X = pd.DataFrame({"a column": [1, 2, 3], "another col": [4, 5, 6]})
    X_expected = pd.DataFrame({"a column": [1], "another col": [4]})
    drop_rows_transformer = DropRowsTransformer(indices_to_drop=[1, 2])
    drop_rows_transformer.fit(X)
    transformed_X = drop_rows_transformer.transform(X)
    assert_frame_equal(X_expected, transformed_X)


def test_drop_rows_transformer_index_not_in_input():
    X = pd.DataFrame({"str": [1, 2]})
    drop_rows_transformer = DropRowsTransformer(indices_to_drop=[1, 2])
    with pytest.raises(ValueError, match="Index does not exist in input DataFrame"):
        drop_rows_transformer.fit(X)


## to test:
## with target specified
## with target not specified


# def test_drop_rows_transformer_string_indices():
# X = pd.DataFrame({"str": [1, 2]})
# drop_rows_transformer = DropRowsTransformer(indices_to_drop=[1, 2])
# with pytest.raises(ValueError, match="Index does not exist in input DataFrame"):
#     drop_rows_transformer.fit(X)
