import pandas as pd
import pytest
from woodwork.logical_types import Boolean, Categorical, Double, Integer

from evalml.pipelines.components import StandardScaler


@pytest.mark.parametrize("X_df", [pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
                                  pd.DataFrame(pd.Series([1., 2., 3.], dtype="float")),
                                  pd.DataFrame(pd.Series([True, False, True], dtype="boolean"))])
def test_standard_scaler_woodwork_custom_overrides_returned_by_components(X_df):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, Boolean]
    for logical_type in override_types:
        X_df.ww.init(logical_types={0: logical_type})

        std_scaler = StandardScaler()
        std_scaler.fit(X_df, y)
        transformed = std_scaler.transform(X_df, y)
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.ww.logical_types == {0: Double}
