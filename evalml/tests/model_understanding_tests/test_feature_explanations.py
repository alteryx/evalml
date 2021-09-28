import pandas as pd

from evalml.model_understanding import explain, get_influential_features

def test_get_influential_features():
    importance_df = pd.DataFrame({'feature': ['heavy influence', 'somewhat influence', 'zero influence', 'negative influence'],
                                  'importance': [0.6, 0.1, 0.0, -0.1]})
    heavy, somewhat, negative = get_influential_features(importance_df, max_features=5)
    assert heavy == ['heavy influence']
    assert somewhat == ['somewhat influence']
    assert negative == ['negative influence']

def test_get_influential_features_max_features():
    importance_df = pd.DataFrame({'feature': ['heavy 1', 'heavy 2', 'heavy 3', 'somewhat 1', 'somewhat 2'],
                                  'importance': [0.35, 0.3, 0.23, 0.15, 0.07]})
    heavy, somewhat, negative = get_influential_features(importance_df, max_features=2)
    assert heavy == ['heavy 1', 'heavy 2']
    assert somewhat == []
    assert negative == []

    heavy, somewhat, negative = get_influential_features(importance_df, max_features=4)
    assert heavy == ['heavy 1', 'heavy 2', 'heavy 3']
    assert somewhat == ['somewhat 1']
    assert negative == []

def test_get_influential_features_max_features_ignore_negative():
    importance_df = pd.DataFrame({'feature': ['heavy 1', 'heavy 2', 'heavy 3', 'neg 1', 'neg 2', 'neg 3'],
                                  'importance': [0.35, 0.3, 0.23, -0.15, -0.17, -0.43]})
    heavy, somewhat, negative = get_influential_features(importance_df, max_features=2)
    assert heavy == ['heavy 1', 'heavy 2']
    assert somewhat == []
    assert negative == ['neg 1', 'neg 2', 'neg 3']
