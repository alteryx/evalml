from evalml.pipelines.components import StandardScaler
import woodwork as ww
import pytest
import pandas as pd
from woodwork.logical_types import Integer, Double, Categorical, NaturalLanguage, Boolean

@pytest.mark.parametrize("logical_type, X_df", [
(ww.logical_types.Integer,pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64"))),
(ww.logical_types.Double, pd.DataFrame(pd.Series([1., 2., 3.], dtype="Float64"))),
(ww.logical_types.Boolean, pd.DataFrame(pd.Series([True, False, True], dtype="boolean")))

])
def test_standard_scaler_woodwork_custom_overrides_returned_by_components(logical_type, X_df):
    y = pd.Series([1, 2, 1])
    types_to_test = [Integer, Double, Categorical, Boolean]
    for l in types_to_test:
        X = None
        override_dict = {0: l}
        try:
            X = ww.DataTable(X_df, logical_types=override_dict)
            assert X.logical_types[0] == l
        except TypeError:
            continue
        print ("testing override", logical_type, "with", l)
        std_scaler = StandardScaler()
        std_scaler.fit(X, y)
        transformed = std_scaler.transform(X, y)
        assert isinstance(transformed, ww.DataTable)
        input_logical_types = {0:l}
        print ("transformed", transformed.logical_types.items())
        print ("expected", input_logical_types.items())
        if l == Categorical:
            assert transformed.logical_types == {0: ww.logical_types.Categorical}
        else:
            assert transformed.logical_types == {0: ww.logical_types.Double}

