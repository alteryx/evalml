from evalml.pipelines import MulticlassClassificationPipeline


def test_multiclass_init():
    clf = MulticlassClassificationPipeline(
        component_graph=["Imputer", "One Hot Encoder", "Random Forest Classifier"],
    )
    assert clf.parameters == {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
            "boolean_impute_strategy": "most_frequent",
            "boolean_fill_value": None,
        },
        "One Hot Encoder": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "Random Forest Classifier": {"n_estimators": 100, "max_depth": 6, "n_jobs": -1},
    }
    assert clf.name == "Random Forest Classifier w/ Imputer + One Hot Encoder"
    assert clf.random_seed == 0
    parameters = {"One Hot Encoder": {"top_n": 20}}
    clf = MulticlassClassificationPipeline(
        component_graph=["Imputer", "One Hot Encoder", "Random Forest Classifier"],
        parameters=parameters,
        custom_name="Custom Pipeline",
        random_seed=42,
    )

    assert clf.parameters == {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
            "boolean_impute_strategy": "most_frequent",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
            "boolean_fill_value": None,
        },
        "One Hot Encoder": {
            "top_n": 20,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "Random Forest Classifier": {"n_estimators": 100, "max_depth": 6, "n_jobs": -1},
    }
    assert clf.name == "Custom Pipeline"
    assert clf.random_seed == 42
