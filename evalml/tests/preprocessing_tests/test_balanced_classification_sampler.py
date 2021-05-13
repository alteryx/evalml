import random

import pandas as pd
import pytest

from evalml.preprocessing.data_splitters import BalancedClassificationSampler


@pytest.mark.parametrize("ratio,samples,percentage,seed",
                         [(1, 1, 0.2, 1),
                          (0.3, 101, 0.5, 100)])
def test_balanced_classification_init(ratio, samples, percentage, seed):
    bcs = BalancedClassificationSampler(sampling_ratio=ratio, min_samples=samples, min_percentage=percentage, random_seed=seed)
    assert bcs.sampling_ratio == ratio
    assert bcs.min_samples == samples
    assert bcs.min_percentage == percentage
    assert bcs.random_seed == seed


def test_balanced_classification_errors():
    with pytest.raises(ValueError, match="sampling_ratio must be"):
        BalancedClassificationSampler(sampling_ratio=1.01)

    with pytest.raises(ValueError, match="sampling_ratio must be"):
        BalancedClassificationSampler(sampling_ratio=-1)

    with pytest.raises(ValueError, match="min_sample must be"):
        BalancedClassificationSampler(min_samples=0)

    with pytest.raises(ValueError, match="min_percentage must be"):
        BalancedClassificationSampler(min_percentage=0)

    with pytest.raises(ValueError, match="min_percentage must be"):
        BalancedClassificationSampler(min_percentage=0.6)

    with pytest.raises(ValueError, match="min_percentage must be"):
        BalancedClassificationSampler(min_percentage=-1.3)


@pytest.mark.parametrize("num_classes", [2, 3])
def test_classification_balanced_simple(num_classes):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([i % num_classes for i in range(1000)])
    bcs = BalancedClassificationSampler()
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)


def test_classification_severely_imbalanced_binary_simple():
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    # 5 instances of positive 1
    y = pd.Series([1 if i % 200 != 0 else 0 for i in range(1000)])
    bcs = BalancedClassificationSampler()
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)


def test_classification_severely_imbalanced_multiclass_simple():
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    # 9 instances of 1, 9 instances of 2
    y = pd.Series([0 if i % 55 != 0 else (1 + i % 2) for i in range(1000)])
    bcs = BalancedClassificationSampler()
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)


@pytest.mark.parametrize("sampling_ratio", [1, 0.5, 0.3, 0.25, 0.2, 0.1])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_classification_imbalanced_sampling_ratio(num_classes, sampling_ratio):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    if num_classes == 2:
        y = pd.Series([0] * 750 + [1] * 250)
    else:
        y = pd.Series([0] * 600 + [1] * 200 + [2] * 200)
    bcs = BalancedClassificationSampler(sampling_ratio=sampling_ratio)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if sampling_ratio <= 1 / 3:
        # the classes are considered balanced, do nothing
        pd.testing.assert_frame_equal(X, X2)
        pd.testing.assert_series_equal(y, y2)
    else:
        # remove some samples
        assert len(X2) == {2: (250 + int(250 / sampling_ratio)), 3: (400 + int(200 / sampling_ratio))}[num_classes]
        assert len(y2) == len(X2)
        assert y2.value_counts().values[0] == int(1 / sampling_ratio) * {2: 250, 3: 200}[num_classes]


@pytest.mark.parametrize("min_samples", [10, 50, 100, 200, 500])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_classification_imbalanced_min_samples(num_classes, min_samples):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    if num_classes == 2:
        y = pd.Series([0] * 900 + [1] * 100)
    else:
        y = pd.Series([0] * 799 + [1] * 101 + [2] * 100)
    bcs = BalancedClassificationSampler(sampling_ratio=1, min_samples=min_samples)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if min_samples <= 100:
        # balance 1:1 without conflicting with min_samples
        assert len(X2) == {2: 200, 3: 300}[num_classes]
        assert y2.value_counts().values[0] == 100
    else:
        # cannot balance 1:1, choosing the min_samples size for the majority class and add minority class(es)
        if num_classes == 2:
            assert len(X2) == min_samples + 100
            assert y2.value_counts().values[0] == min_samples
        else:
            assert len(X2) == min_samples + 201
            assert y2.value_counts().values[0] == min_samples


@pytest.mark.parametrize("min_percentage", [0.01, 0.05, 0.2, 0.3])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_classification_imbalanced_min_percentage(num_classes, min_percentage):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    if num_classes == 2:
        y = pd.Series([0] * 950 + [1] * 50)
    else:
        y = pd.Series([0] * 820 + [1] * 90 + [2] * 90)
    bcs = BalancedClassificationSampler(sampling_ratio=1, min_percentage=min_percentage)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if min_percentage <= 0.05:
        # does not classify as severe imbalance, so balance 1:1 with min_samples==100
        assert len(X2) == {2: 150, 3: 280}[num_classes]
        assert y2.value_counts().values[0] == 100
    else:
        # severe imbalance, do nothing
        pd.testing.assert_frame_equal(X2, X)


@pytest.mark.parametrize("min_percentage", [0.01, 0.05, 0.2, 0.3])
@pytest.mark.parametrize("min_samples", [10, 50, 100, 200, 500])
def test_classification_imbalanced_severe_imbalance_binary(min_samples, min_percentage):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([0] * 850 + [1] * 150)  # minority class is 15% of total distribution
    bcs = BalancedClassificationSampler(sampling_ratio=0.5, min_samples=min_samples, min_percentage=min_percentage)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if min_samples >= 200 and min_percentage >= 0.2:
        # severe imbalance, do nothing
        pd.testing.assert_frame_equal(X2, X)
    else:
        # does not classify as severe imbalance, so balance 2:1 with min_samples
        assert len(X2) == 150 + max(min_samples, 2 * 150)
        assert y2.value_counts().values[0] == max(min_samples, 2 * 150)


@pytest.mark.parametrize("sampling_ratio", [1, 0.5, 0.33, 0.25, 0.2, 0.1])
@pytest.mark.parametrize("min_samples", [10, 50, 100, 200, 500])
def test_classification_imbalanced_normal_imbalance_binary(min_samples, sampling_ratio):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([0] * 850 + [1] * 150)  # minority class is 15% of total distribution, never counts as severe imbalance
    bcs = BalancedClassificationSampler(sampling_ratio=sampling_ratio, min_samples=min_samples)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if sampling_ratio < 0.2:
        # data is balanced, do nothing
        pd.testing.assert_frame_equal(X2, X)
    else:
        # rebalance according to the ratio and min_samples
        assert len(X2) == 150 + max(min_samples, int(150 / sampling_ratio))
        assert y2.value_counts().values[0] == max(min_samples, int(150 / sampling_ratio))


@pytest.mark.parametrize("data_type", ['n', 's'])
@pytest.mark.parametrize("min_percentage", [0.01, 0.05, 0.2, 0.3])
@pytest.mark.parametrize("min_samples", [10, 50, 100, 200, 500])
def test_classification_imbalanced_severe_imbalance_multiclass(data_type, min_samples, min_percentage):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    if data_type == 'n':
        y = pd.Series([0] * 800 + [1] * 100 + [2] * 100)  # minority class is 10% of total distribution
    else:
        y = pd.Series(["class_1"] * 800 + ["class_2"] * 100 + ["class_3"] * 100)
    bcs = BalancedClassificationSampler(sampling_ratio=0.5, min_samples=min_samples, min_percentage=min_percentage)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if min_samples >= 200 and min_percentage >= 0.2:
        # severe imbalance, do nothing
        pd.testing.assert_frame_equal(X2, X)
    else:
        # does not classify as severe imbalance, so balance 2:1 with min_samples
        assert len(X2) == 200 + max(min_samples, 2 * 100)
        assert y2.value_counts().values[0] == max(min_samples, 2 * 100)


@pytest.mark.parametrize("data_type", ['n', 's'])
@pytest.mark.parametrize("sampling_ratio", [1, 0.5, 0.33, 0.25, 0.2, 0.1])
@pytest.mark.parametrize("min_samples", [10, 50, 100, 200, 500])
def test_classification_imbalanced_normal_imbalance_multiclass(data_type, min_samples, sampling_ratio):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    if data_type == 'n':
        y = pd.Series([0] * 800 + [1] * 100 + [2] * 100)  # minority class is 10% of total distribution
    else:
        y = pd.Series(["class_1"] * 800 + ["class_2"] * 100 + ["class_3"] * 100)
    bcs = BalancedClassificationSampler(sampling_ratio=sampling_ratio, min_samples=min_samples)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if sampling_ratio < 0.2:
        # data is balanced, do nothing
        pd.testing.assert_frame_equal(X2, X)
    else:
        # rebalance according to the ratio and min_samples
        assert len(X2) == 200 + max(min_samples, int(100 / sampling_ratio))
        assert y2.value_counts().values[0] == max(min_samples, int(100 / sampling_ratio))


@pytest.mark.parametrize("sampling_ratio", [1, 0.5, 0.33, 0.25, 0.2, 0.1])
@pytest.mark.parametrize("random_seed", [0, 1, 2, 300])
def test_classification_imbalanced_random_seed(random_seed, sampling_ratio):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([0] * 800 + [1] * 200)
    bcs1 = BalancedClassificationSampler(sampling_ratio=sampling_ratio, random_seed=random_seed)
    bcs2 = BalancedClassificationSampler(sampling_ratio=sampling_ratio, random_seed=random_seed)
    indices1 = bcs1.fit_resample(X, y)
    X1 = X.loc[indices1]
    y1 = y.loc[indices1]
    indices2 = bcs2.fit_resample(X, y)
    X2 = X.loc[indices2]
    y2 = y.loc[indices2]

    if sampling_ratio <= 0.25:
        # data is balanced
        pd.testing.assert_frame_equal(X1, X)
    else:
        assert len(X2) == 200 + int(200 / sampling_ratio)
        assert y2.value_counts().values[0] == int(200 / sampling_ratio)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


@pytest.mark.parametrize("index", [[f'hello_{i}' for i in range(1000)],
                                   random.shuffle([i + 0.5 for i in range(1000)]),
                                   pd.MultiIndex.from_arrays([
                                       [f"index_{i}" for i in range(1000)],
                                       [i for i in range(1000)]
                                   ])])
def test_classification_imbalanced_custom_indices(index):
    X = pd.DataFrame({"a": [i for i in range(1000)]}, index=index)
    y = pd.Series([0] * 900 + [1] * 100, index=index)
    bcs = BalancedClassificationSampler()
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    assert len(X2) == 500
    assert all(y2.value_counts(0).values == [400, 100])
    assert all(y2.index.values == X2.index.values)
    assert len(set(y2.index.values).intersection(set(y.index.values))) == len(y2)


@pytest.mark.parametrize("size", [100, 200, 500])
def test_classification_imbalanced_small_dataset(size):
    X = pd.DataFrame({"a": [i for i in range(size)]})
    y = pd.Series([0] * int(0.8 * size) + [1] * int(0.2 * size))
    bcs = BalancedClassificationSampler(sampling_ratio=1)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if size == 100:
        pd.testing.assert_frame_equal(X2, X)
    else:
        assert len(X2) == 0.2 * size + 100

    bcs2 = BalancedClassificationSampler(sampling_ratio=1, min_samples=40)
    indices = bcs2.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    if size == 500:
        # resulting majority size is 100
        assert len(X2) == 200
        assert y2.value_counts(normalize=True).values[0] == 0.5
    else:
        assert len(X2) == 0.2 * size + 40
        assert y2.value_counts().values[0] == 40


def test_classification_imbalanced_multiple_multiclass():
    X = pd.DataFrame({"a": [i for i in range(10000)]})
    y = pd.Series([0] * 4900 + [1] * 4900 + [2] * 200)  # minority class is 2% of data
    bcs = BalancedClassificationSampler(min_samples=201)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    # severe imbalanace case, don't resample
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)

    bcs = BalancedClassificationSampler()
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    assert len(X2) == 1800
    assert all(y2.value_counts().values == [800, 800, 200])
    assert y2.value_counts()[2] == 200

    bcs = BalancedClassificationSampler(sampling_ratio=.3333)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    # resample to 4:1 ratios on both 0 and 1 classes
    assert len(X2) == 1400
    assert all(y2.value_counts().values == [600, 600, 200])
    assert y2.value_counts()[2] == 200


@pytest.mark.parametrize("data_type", ['li', 'np', 'pd', 'ww'])
def test_classification_imbalanced_data_type(data_type, make_data_type):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([0] * 900 + [1] * 100)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    bcs = BalancedClassificationSampler()
    indices = bcs.fit_resample(X, y)
    assert len(indices) == 500
    if data_type in ['pd', 'np']:
        y2 = y.loc[indices]
        assert all(y2.value_counts().values == [400, 100])
        assert y2.value_counts()[1] == 100


def test_classification_data_frame_dtypes():
    X = pd.DataFrame({
        "integers": [i for i in range(1000)],
        "strings": [f"string_{i % 3}" for i in range(1000)],
        "text": [f"this should be text data because {i} think it's a long string. Let's hope it behaves in that way" for i in range(1000)],
        "float": [i / 10000 for i in range(1000)],
        "bool": [bool(i % 2) for i in range(1000)],
        "datetime": [random.choice([2012 / 1 / 2, 2012 / 2 / 1, 2012 / 4 / 2]) for i in range(1000)]
    })
    y = pd.Series([0] * 900 + [1] * 100)
    bcs = BalancedClassificationSampler()
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    assert len(X2) == 500
    assert all(y2.value_counts().values == [400, 100])
    assert y2.value_counts()[1] == 100

    X['integers'][0] = None
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    assert len(X2) == 500
    assert all(y2.value_counts().values == [400, 100])
    assert y2.value_counts()[1] == 100


def test_classification_data_drop():
    # tests for whether or not the `max(0, counts[k] - goal_value)` code works as expected
    X = pd.DataFrame({"a": [i for i in range(420)]})
    y = pd.Series([0] * 90 + [1] * 100 + [2] * 120 + [3] * 40 + [4] * 70)
    # will downsample the [2] target
    # will try to downsample [0] and [4], but max(0, x) will prevent that
    bcs = BalancedClassificationSampler(sampling_ratio=1, min_percentage=0.01)
    indices = bcs.fit_resample(X, y)
    X2 = X.loc[indices]
    y2 = y.loc[indices]
    assert len(X2) == 400
    assert y2.value_counts().values[0] == 100


def test_balance_ratio_value():
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([0] * 200 + [1] * 800)
    bcs = BalancedClassificationSampler(sampling_ratio=0.1)
    indices = bcs.fit_resample(X, y)
    # make sure there was no resampling done
    assert len(indices) == 1000


def test_dict_overrides_ratio():
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([0] * 200 + [1] * 800)
    sampling_ratio_dict = {0: 200, 1: 800}
    bcs = BalancedClassificationSampler(sampling_ratio=0.1, sampling_ratio_dict=sampling_ratio_dict)
    indices = bcs.fit_resample(X, y)
    y_new = y.iloc[indices]
    y_sampled_count = y_new.value_counts().to_dict()
    assert y_sampled_count == sampling_ratio_dict


@pytest.mark.parametrize("sampling_ratio_dict,expected", [({0: 200, 1: 700}, {0: 200, 1: 700}),
                                                          ({0: 100, 1: 100}, {0: 100, 1: 100}),
                                                          ({0: 200, 1: 800}, {0: 200, 1: 800}),
                                                          ({0: 100, 1: 805}, {0: 100, 1: 800}),
                                                          ({0: 200, 1: 805}, {0: 200, 1: 800})])
def test_sampler_ratio_dictionary_binary(sampling_ratio_dict, expected):
    X = pd.DataFrame({"a": [i for i in range(1000)]})
    y = pd.Series([0] * 200 + [1] * 800)
    bcs = BalancedClassificationSampler(sampling_ratio_dict=sampling_ratio_dict)
    indices = bcs.fit_resample(X, y)
    y_new = y.iloc[indices]
    y_sampled_count = y_new.value_counts().to_dict()
    assert y_sampled_count == expected


@pytest.mark.parametrize("sampling_ratio_dict,expected", [({0: 200, 1: 700, 2: 150}, {0: 200, 1: 700, 2: 150}),
                                                          ({0: 100, 1: 100, 2: 150}, {0: 100, 1: 100, 2: 150}),
                                                          ({0: 200, 1: 800, 2: 200}, {0: 200, 1: 800, 2: 200}),
                                                          ({0: 100, 1: 805, 2: 400}, {0: 100, 1: 800, 2: 200}),
                                                          ({0: 200, 1: 805, 2: 400}, {0: 200, 1: 800, 2: 200})])
def test_sampler_ratio_dictionary_multiclass(sampling_ratio_dict, expected):
    X = pd.DataFrame({"a": [i for i in range(1200)]})
    y = pd.Series([0] * 200 + [1] * 800 + [2] * 200)
    bcs = BalancedClassificationSampler(sampling_ratio_dict=sampling_ratio_dict)
    indices = bcs.fit_resample(X, y)
    y_new = y.iloc[indices]
    y_sampled_count = y_new.value_counts().to_dict()
    assert y_sampled_count == expected
