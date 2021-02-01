import numpy as np
import pytest
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, StratifiedKFold

from evalml.preprocessing.data_splitters import SMOTETomekTVSplit, SMOTETomekCVSplit
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)

def test_kmeans_smote_nsplits():
    assert SMOTETomekTVSplit().get_n_splits() == 1
    assert SMOTETomekCVSplit().get_n_splits() == 3
    assert SMOTETomekCVSplit(n_splits=5).get_n_splits() == 5


@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
def test_smote_tomek_tv_default(data_type, make_data_type, X_y_binary):
    X, y = X_y_binary

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    st = SMOTETomek(random_state=0)
    X_resample, y_resample = st.fit_resample(X_train, y_train)
    initial_results = [(X_resample, y_resample), (X_test, y_test)]
    X_transform, y_transform = st.fit_resample(X, y)

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    stsplit = SMOTETomekTVSplit(random_state=0, test_size=0.2)
    for i, j in enumerate(stsplit.split(X, y)):
        for idx, tup in enumerate(j):
            for jdx, val in enumerate(tup):
                if jdx == 1:
                    np.testing.assert_equal(val.to_series().values, initial_results[idx][jdx])
                else:
                    np.testing.assert_equal(val.to_dataframe().values, initial_results[idx][jdx])

    X_stsplit, y_stsplit = stsplit.transform(X, y)
    np.testing.assert_equal(X_transform, X_stsplit.to_dataframe().values)
    np.testing.assert_equal(y_transform, y_stsplit.to_series().values)


@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
def test_smote_tomek_cv_default(data_type, make_data_type, X_y_binary):
    X, y = X_y_binary
    skf = StratifiedKFold(shuffle=True, n_splits=3, random_state=0)
    st = SMOTETomek(random_state=0)
    initial_results = []
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        X_resample, y_resample = st.fit_resample(X[train_indices], y[train_indices])
        initial_results.append([(X_resample, y_resample), (X[test_indices], y[test_indices])])
    X_transform, y_transform = st.fit_resample(X, y)

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    stsplit = SMOTETomekCVSplit(random_state=0)
    for i, j in enumerate(stsplit.split(X, y)):
        for idx, tup in enumerate(j):
            for jdx, val in enumerate(tup):
                if jdx == 1:
                    np.testing.assert_equal(val.to_series().values, initial_results[i][idx][jdx])
                else:
                    np.testing.assert_equal(val.to_dataframe().values, initial_results[i][idx][jdx])

    X_stsplit, y_stsplit = stsplit.transform(X, y)
    np.testing.assert_equal(X_transform, X_stsplit.to_dataframe().values)
    np.testing.assert_equal(y_transform, y_stsplit.to_series().values)
