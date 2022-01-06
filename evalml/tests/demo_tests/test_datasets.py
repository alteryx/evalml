import urllib

import pandas as pd
import pytest

from evalml import demos


@pytest.fixture(autouse=True, scope="session")
def set_testing_headers():
    opener = urllib.request.build_opener()
    opener.addheaders = [("Testing", "True")]
    urllib.request.install_opener(opener)


@pytest.fixture(autouse=True, scope="session")
def check_online(set_testing_headers):
    try:
        urllib.request.urlopen("https://api.featurelabs.com/update_check/")
        return True
    except urllib.error.URLError:  # pragma: no cover
        return False


@pytest.fixture(autouse=True)
def skip_offline(request, check_online):
    if (
        request.node.get_closest_marker("skip_offline") and not check_online
    ):  # pragma: no cover
        pytest.skip("Cannot reach update server, skipping online tests")


@pytest.fixture
def local_datasets(
    fraud_local,
    wine_local,
    churn_local,
    daily_temp_local,
    breast_cancer_local,
    diabetes_local,
):
    local_datasets = {
        "fraud": fraud_local,
        "wine": wine_local,
        "churn": churn_local,
        "daily_temp": daily_temp_local,
        "breast_cancer": breast_cancer_local,
        "diabetes": diabetes_local,
    }
    return local_datasets


@pytest.mark.parametrize(
    "dataset_name, expected_shape",
    [
        ("fraud", (99992, 12)),
        ("wine", (178, 13)),
        ("breast_cancer", (569, 30)),
        ("diabetes", (442, 10)),
        ("churn", (7043, 19)),
        ("daily_temp", (3650, 1)),
    ],
)
def test_datasets(dataset_name, expected_shape, local_datasets):
    X, y = local_datasets[dataset_name]
    assert X.shape == expected_shape
    assert y.shape == (expected_shape[0],)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.ww.schema is not None
    assert y.ww.schema is not None


@pytest.mark.parametrize(
    "dataset_name, demo_method",
    [
        ("fraud", demos.load_fraud()),
        ("wine", demos.load_wine()),
        ("breast_cancer", demos.load_breast_cancer()),
        ("diabetes", demos.load_diabetes()),
        ("churn", demos.load_churn()),
        ("daily_temp", demos.load_weather()),
    ],
)
@pytest.mark.skip_offline
def test_datasets_match_local(dataset_name, demo_method, local_datasets):
    X, y = demo_method
    X_local, y_local = local_datasets[dataset_name]

    if dataset_name == "daily_temp":
        missing_date_1 = pd.DataFrame([pd.to_datetime("1984-12-31")], columns=["Date"])
        missing_date_2 = pd.DataFrame([pd.to_datetime("1988-12-31")], columns=["Date"])
        missing_y_1 = pd.Series([14.5], name="Temp")
        missing_y_2 = pd.Series([14.5], name="Temp")

        X_local = pd.concat(
            [
                X_local.iloc[:1460],
                missing_date_1,
                X_local.iloc[1460:2920],
                missing_date_2,
                X_local.iloc[2920:],
            ]
        ).reset_index(drop=True)
        y_local = pd.concat(
            [
                y_local.iloc[:1460],
                missing_y_1,
                y_local.iloc[1460:2920],
                missing_y_2,
                y_local.iloc[2920:],
            ]
        ).reset_index(drop=True)

    pd.testing.assert_frame_equal(X, X_local)
    pd.testing.assert_series_equal(y, y_local)
