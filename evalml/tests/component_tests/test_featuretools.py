from unittest.mock import patch

import featuretools as ft
import pandas as pd
import pytest
import woodwork as ww

from evalml.pipelines.components import DFSTransformer


def test_index_errors(X_y_binary):
    with pytest.raises(TypeError, match="Index provided must be string"):
        DFSTransformer(index=0)

    with pytest.raises(TypeError, match="Index provided must be string"):
        DFSTransformer(index=None)


def test_numeric_columns(X_y_multi):
    X, y = X_y_multi
    X_pd = pd.DataFrame(X)

    feature = DFSTransformer()
    feature.fit(X_pd, y)
    feature.transform(X_pd)


@patch('evalml.pipelines.components.transformers.preprocessing.featuretools.dfs')
@patch('evalml.pipelines.components.transformers.preprocessing.featuretools.calculate_feature_matrix')
def test_index(mock_dfs, mock_calculate_feature_matrix, X_y_multi):
    X, y = X_y_multi
    X_pd = pd.DataFrame(X)
    X_new_index = X_pd.copy()
    index = [i for i in range(len(X))]
    new_index = [i * 2 for i in index]
    X_new_index['index'] = new_index

    # check if _make_entity_set keeps the intended index
    feature = DFSTransformer()
    feature.fit(X_new_index)
    feature.transform(X_new_index)
    arg_es = mock_dfs.call_args[1]['entityset'].entities[0].df['index']
    arg_tr = mock_calculate_feature_matrix.call_args[1]['entityset'].entities[0].df['index']
    assert arg_es.to_list() == new_index
    assert arg_tr.to_list() == new_index

    # check if _make_entity_set fills in the proper index values
    feature.fit(X_pd)
    feature.transform(X_pd)
    arg_es = mock_dfs.call_args[1]['entityset'].entities[0].df['index']
    arg_tr = mock_calculate_feature_matrix.call_args[1]['entityset'].entities[0].df['index']
    assert arg_es.to_list() == index
    assert arg_tr.to_list() == index


def test_transform(X_y_binary, X_y_multi, X_y_regression):
    datasets = locals()
    for dataset in datasets.values():
        X, y = dataset
        X_pd = pd.DataFrame(X)
        X_pd.columns = X_pd.columns.astype(str)
        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id="X", dataframe=X_pd, index='index', make_index=True)
        matrix, features = ft.dfs(entityset=es, target_entity="X")

        feature = DFSTransformer()
        feature.fit(X)
        X_feature_matrix = feature.transform(X)

        pd.testing.assert_frame_equal(matrix, X_feature_matrix)
        assert features == feature.features

        feature.fit(X, y)
        feature.transform(X)

        X_ww = ww.DataTable(X_pd)
        feature.fit(X_ww)
        feature.transform(X_ww)


def test_transform_subset(X_y_binary, X_y_multi, X_y_regression):
    datasets = locals()
    for dataset in datasets.values():
        X, y = dataset
        X_pd = pd.DataFrame(X)
        X_pd.columns = X_pd.columns.astype(str)
        X_fit = X_pd.iloc[: len(X) // 3]
        X_transform = X_pd.iloc[len(X) // 3:]

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id="X", dataframe=X_transform, index='index', make_index=True)
        matrix, features = ft.dfs(entityset=es, target_entity="X")

        feature = DFSTransformer()
        feature.fit(X_fit)
        X_feature_matrix = feature.transform(X_transform)

        pd.testing.assert_frame_equal(matrix, X_feature_matrix)
