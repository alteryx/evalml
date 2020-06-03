import pandas as pd
import pytest

from evalml.pipelines.components import DropNullColumns


def test_drop_null_transformer_init():
    drop_null_transformer = DropNullColumns()
    assert drop_null_transformer.parameters["pct_null_threshold"] == 1.0
    assert drop_null_transformer.cols_to_drop is None

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.95)
    assert drop_null_transformer.parameters["pct_null_threshold"] == 0.95
    assert drop_null_transformer.cols_to_drop is None

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DropNullColumns(pct_null_threshold=-0.95)

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DropNullColumns(pct_null_threshold=1.01)


def test_drop_null_transformer_without_fit():
    drop_null_transformer = DropNullColumns()
    with pytest.raises(RuntimeError):
        drop_null_transformer.transform(pd.DataFrame())
