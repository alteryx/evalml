import pandas as pd
import pytest
import woodwork as ww
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Double,
    Integer,
    NaturalLanguage
)

from evalml.pipelines.components import StandardScaler


@pytest.mark.parametrize("logical_type, X_df", [
(ww.logical_types.Integer, pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64"))),
(ww.logical_types.Double, pd.DataFrame(pd.Series([1., 2., 3.], dtype="float"))),
(ww.logical_types.Boolean, pd.DataFrame(pd.Series([True, False, True], dtype="boolean")))

])
def test_standard_scaler_woodwork_custom_overrides_returned_by_components(logical_type, X_df):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, Boolean]
    for l in override_types:
        X = None
        override_dict = {0: l}
        try:
            X = ww.DataTable(X_df, logical_types=override_dict)
            assert X.logical_types[0] == l
        except TypeError:
            continue

        std_scaler = StandardScaler()
        std_scaler.fit(X, y)
        transformed = std_scaler.transform(X, y)
        assert isinstance(transformed, ww.DataTable)
        input_logical_types = {0: l}

        if l == Categorical:
            assert transformed.logical_types == {0: ww.logical_types.Categorical}
        else:
            assert transformed.logical_types == {0: ww.logical_types.Double}
