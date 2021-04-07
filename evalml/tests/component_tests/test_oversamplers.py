import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import (
    SMOTENCSampler,
    SMOTENSampler,
    SMOTESampler
)
from evalml.pipelines.components.utils import make_balancing_dictionary

im = pytest.importorskip('imblearn.over_sampling', reason='Skipping test because imbalanced-learn not installed')


binary = pd.Series([0] * 800 + [1] * 200)
multiclass = pd.Series([0] * 800 + [1] * 150 + [2] * 50)


@pytest.mark.parametrize("y,sampling_ratio,result",
                         [(binary, 1, {0: 800, 1: 800}),
                          (binary, 0.5, {0: 800, 1: 400}),
                          (binary, 0.25, {0: 800, 1: 200}),
                          (binary, 0.1, {0: 800, 1: 200}),
                          (multiclass, 1, {0: 800, 1: 800, 2: 800}),
                          (multiclass, 0.5, {0: 800, 1: 400, 2: 400}),
                          (multiclass, 0.25, {0: 800, 1: 200, 2: 200}),
                          (multiclass, 0.1, {0: 800, 1: 150, 2: 80})])
def test_make_balancing_dictionary(y, sampling_ratio, result):
    dic = make_balancing_dictionary(y, sampling_ratio)
    assert dic == result


@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
def test_init(sampler):
    parameters = {
        "sampling_ratio": 0.5,
        "sampling_ratio_dict": {},
        "k_neighbors": 2,
        "n_jobs": -1
    }
    if 'SMOTENC' in sampler.name:
        parameters['categorical_features'] = [0]
    oversampler = sampler(**parameters)
    assert oversampler.parameters == parameters


@pytest.mark.parametrize("cat_cols,fails", [([], True),
                                            ([False], True),
                                            ([i for i in range(20)], True),
                                            ([True for i in range(20)], True),
                                            ([1, 2, 3], False),
                                            ([True for i in range(19)] + [False], False),
                                            ([0, 1, 9], False)])
def test_smotenc_fails(cat_cols, fails, X_y_binary):
    # X has 20 columns
    X, y = X_y_binary
    if fails:
        with pytest.raises(ValueError, match='categorical_features must be longer'):
            SMOTENCSampler(sampling_ratio=1, categorical_features=cat_cols).fit_transform(X, y)
    else:
        SMOTENCSampler(sampling_ratio=1, categorical_features=cat_cols).fit_transform(X, y)


@pytest.mark.parametrize("sampler", [SMOTESampler(sampling_ratio=1),
                                     SMOTENCSampler(sampling_ratio=1, categorical_features=[0]),
                                     SMOTENSampler(sampling_ratio=1)])
def test_none_y(sampler):
    X = pd.DataFrame([[i] for i in range(5)])
    oversampler = sampler
    with pytest.raises(ValueError, match="y cannot be none"):
        oversampler.fit(X, None)
    with pytest.raises(ValueError, match="y cannot be none"):
        oversampler.fit_transform(X, None)
    oversampler.fit(X, pd.Series([0] * 4 + [1]))
    oversampler.transform(X, None)


@pytest.mark.parametrize("sampler", [SMOTESampler(sampling_ratio=1),
                                     SMOTENCSampler(sampling_ratio=1, categorical_features=[0]),
                                     SMOTENSampler(sampling_ratio=1)])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_no_oversample(data_type, sampler, make_data_type, X_y_binary):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler
    new_X, new_y = oversampler.fit_transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, new_X.to_dataframe().values)
    np.testing.assert_equal(y, new_y.to_series().values)


@pytest.mark.parametrize("sampler", [SMOTESampler(sampling_ratio=1),
                                     SMOTENCSampler(sampling_ratio=1, categorical_features=[1]),
                                     SMOTENSampler(sampling_ratio=1)])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_binary(data_type, sampler, make_data_type):
    X = np.array([[i for i in range(1000)],
                  [i % 7 for i in range(1000)],
                  [0.3 * (i % 3) for i in range(1000)]]).T
    y = np.array([0] * 150 + [1] * 850)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler
    new_X, new_y = oversampler.fit_transform(X, y)

    new_length = 1700
    assert len(new_X) == new_length
    assert len(new_y) == new_length
    value_counts = new_y.to_series().value_counts()
    assert value_counts.values[0] == value_counts.values[1]
    pd.testing.assert_series_equal(value_counts, pd.Series([850, 850]), check_dtype=False)

    transform_X, transform_y = oversampler.transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, transform_X.to_dataframe().values)
    np.testing.assert_equal(y, transform_y.to_series().values)


@pytest.mark.parametrize("sampling_ratio_dict", [{}, {0: 800, 1: 300, 2: 300}])
@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_multiclass(data_type, sampler, sampling_ratio_dict, make_data_type):
    X = np.array([[i for i in range(1000)],
                  [i % 7 for i in range(1000)],
                  [0.3 * (i % 3) for i in range(1000)]]).T
    y = np.array([0] * 800 + [1] * 100 + [2] * 100)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler(sampling_ratio_dict=sampling_ratio_dict)
    if 'NC' in sampler.name:
        oversampler = sampler(categorical_features=[1], sampling_ratio_dict=sampling_ratio_dict)

    new_X, new_y = oversampler.fit_transform(X, y)

    if sampling_ratio_dict == {}:
        new_length = 1200
        num_samples = [800, 200, 200]
    else:
        new_length = sum(sampling_ratio_dict.values())
        num_samples = [800, 300, 300]
    # check the lengths and sampled values are as we expect
    assert len(new_X) == new_length
    assert len(new_y) == new_length
    value_counts = new_y.to_series().value_counts()
    assert value_counts.values[1] == value_counts.values[2]
    np.testing.assert_equal(value_counts.values, np.array(num_samples))

    transform_X, transform_y = oversampler.transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, transform_X.to_dataframe().values)
    np.testing.assert_equal(y, transform_y.to_series().values)


@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
def test_oversample_seed_same_outputs(sampler, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series([0] * 90 + [1] * 10)

    samplers = []
    for seed in [0, 0, 1]:
        oversampler = sampler(sampling_ratio=1, random_seed=seed)
        if 'NC' in sampler.name:
            oversampler = sampler(categorical_features=[1], sampling_ratio=1, random_seed=seed)
        samplers.append(oversampler)

    # iterate through different indices in samplers
    # in group 1, first two oversamplers in samplers should be equal
    # in group 2, calling same oversamplers twice should be equal
    # in group 3, last two oversamplers in samplers should be different
    for s1, s2 in [[0, 1], [1, 1], [1, 2]]:
        X1, y1 = samplers[s1].fit_transform(X, y)
        X2, y2 = samplers[s2].fit_transform(X, y)
        if s2 == 2 and sampler != SMOTENSampler:
            # group 3, SMOTENSampler performance doesn't change with different random states
            with pytest.raises(AssertionError):
                pd.testing.assert_frame_equal(X1.to_dataframe(), X2.to_dataframe())
        else:
            pd.testing.assert_frame_equal(X1.to_dataframe(), X2.to_dataframe())
        pd.testing.assert_series_equal(y1.to_series(), y2.to_series())


@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
def test_sampler_dic_overrides_ratio(sampler):
    X = np.array([[i for i in range(1000)],
                  [i % 7 for i in range(1000)],
                  [0.3 * (i % 3) for i in range(1000)]]).T
    y = np.array([0] * 800 + [1] * 100 + [2] * 100)

    goal_value_dic = {0: 800, 1: 800, 2: 600}
    oversampler = sampler(sampling_ratio=1, sampling_ratio_dict=goal_value_dic)
    if 'NC' in sampler.name:
        oversampler = sampler(categorical_features=[1], sampling_ratio=1, sampling_ratio_dict=goal_value_dic)

    new_X, new_y = oversampler.fit_transform(X, y)
    assert len(new_X) == sum(goal_value_dic.values())
    assert len(new_y) == sum(goal_value_dic.values())


@pytest.mark.parametrize("component_sampler,imblearn_sampler",
                         [(SMOTESampler, im.SMOTE),
                          (SMOTENCSampler, im.SMOTENC),
                          (SMOTENSampler, im.SMOTEN)])
@pytest.mark.parametrize("problem_type", ['binary', 'multiclass'])
def test_samplers_perform_equally(problem_type, component_sampler, imblearn_sampler, X_y_binary, X_y_multi):
    if problem_type == 'binary':
        X, _ = X_y_binary
        y = np.array([0] * 90 + [1] * 10)
        sampling_ratio = 0.5
        sampling_dic = {'sampling_ratio': sampling_ratio}
        expected_y = np.array([0] * 90 + [1] * 45)
    else:
        X, _ = X_y_multi
        y = np.array([0] * 70 + [1] * 20 + [2] * 10)
        sampling_ratio = {0: 70, 1: 40, 2: 20}
        sampling_dic = {'sampling_ratio_dict': sampling_ratio}
        expected_y = np.array([0] * 70 + [1] * 40 + [2] * 20)

    random_seed = 1
    if component_sampler != SMOTENCSampler:
        component = component_sampler(**sampling_dic, random_seed=random_seed)
        imb_sampler = imblearn_sampler(sampling_strategy=sampling_ratio, random_state=random_seed)
    else:
        component = component_sampler(**sampling_dic, categorical_features=[1, 2, 3, 4], random_seed=random_seed)
        imb_sampler = imblearn_sampler(sampling_strategy=sampling_ratio, categorical_features=[1, 2, 3, 4], random_state=random_seed)

    X_com, y_com = component.fit_transform(X, y)
    X_im, y_im = imb_sampler.fit_resample(X, y)

    np.testing.assert_equal(X_com.to_dataframe().values, X_im)
    np.testing.assert_equal(y_com.to_series().values, y_im)
    np.testing.assert_equal(sorted(y_im), expected_y)
