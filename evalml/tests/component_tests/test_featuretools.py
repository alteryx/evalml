from unittest.mock import patch

import featuretools as ft
import pandas as pd
import pytest
import woodwork as ww

from evalml.pipelines.components import FeaturetoolsTransformer


def test_index_errors(X_y_binary):
    with pytest.raises(TypeError, match="Index provided must be string"):
        FeaturetoolsTransformer(index=0)

    with pytest.raises(TypeError, match="Index provided must be string"):
        FeaturetoolsTransformer(index=None)

    with pytest.raises(TypeError, match="Target entity provided must be string, got "):
        FeaturetoolsTransformer(target_entity=0)

    X = pd.DataFrame()
    with pytest.raises(TypeError, match="Target entity provided must be string, got "):
        FeaturetoolsTransformer(target_entity=X)


def test_numeric_columns(X_y_multi):
    X, y = X_y_multi
    X_pd = pd.DataFrame(X)

    feature = FeaturetoolsTransformer()
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
    feature = FeaturetoolsTransformer()
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
        es = es.entity_from_dataframe(entity_id="X_name", dataframe=X_pd, index='index', make_index=True)
        matrix, features = ft.dfs(entityset=es, target_entity="X_name")

        feature = FeaturetoolsTransformer(target_entity="X_name")
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

        feature = FeaturetoolsTransformer()
        feature.fit(X_fit)
        X_feature_matrix = feature.transform(X_transform)

        pd.testing.assert_frame_equal(matrix, X_feature_matrix)


def test_single_entityset(X_y_binary, X_y_multi, X_y_regression):
    datasets = locals()
    for dataset in datasets.values():
        X, y = dataset
        X_pd = pd.DataFrame(X)
        X_pd.columns = X_pd.columns.astype(str)

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id="X", dataframe=X_pd, index='index', make_index=True)
        matrix, features = ft.dfs(entityset=es, target_entity="X")

        feature = FeaturetoolsTransformer()
        feature.fit(es)
        transformed_es = feature.transform(es)

        pd.testing.assert_frame_equal(matrix, transformed_es)


def test_multiple_entityset(X_y_binary, X_y_multi, X_y_regression):
    datasets = locals()
    user_df = pd.DataFrame({"name": ["a", "b", "c", "d", "e"], "id": [0, 1, 2, 3, 4], "extra info": [True, False, False, True, True]})
    for dataset in datasets.values():
        X, y = dataset
        X_pd = pd.DataFrame(X)
        X_pd.columns = X_pd.columns.astype(str)
        # add fake ids to the columns
        X_pd['ids'] = [i % 5 for i in range(len(X))]

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id="X_initial", dataframe=X_pd, index='index', make_index=True)
        es = es.entity_from_dataframe(entity_id="X_users", dataframe=user_df, index='id')
        new_relationship = ft.Relationship(es['X_users']['id'], es['X_initial']['ids'])
        es.add_relationship(new_relationship)
        matrix, features = ft.dfs(entityset=es, target_entity="X_initial")

        feature = FeaturetoolsTransformer(target_entity="X_initial")
        feature.fit(es)
        transformed_es = feature.transform(es)

        pd.testing.assert_frame_equal(matrix, transformed_es)


def test_entity_name(X_y_binary):
    X, y = X_y_binary
    X_pd = pd.DataFrame(X)
    X_pd.columns = X_pd.columns.astype(str)

    feature = FeaturetoolsTransformer(target_entity="X_initial")
    feature.fit(X_pd)

    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id="X_initial", dataframe=X_pd, index='index', make_index=True)

    with pytest.raises(KeyError, match="Provided target entity X_name does not exist in entity"):
        FeaturetoolsTransformer(target_entity="X_name").fit(es)

    feature = FeaturetoolsTransformer(target_entity="X_initial")
    feature.fit(es)
    feature.transform(es)
