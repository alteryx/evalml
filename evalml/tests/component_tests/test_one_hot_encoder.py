import numpy as np
import pandas as pd

from evalml.pipelines.components import OneHotEncoder

def test_null_values():
    X = pd.DataFrame([[2, 0, 1, 0], [np.nan,"b","a","c"]])
    with pytest.raises(ValueError, match="Dataframe to be encoded can not contain null values."):
