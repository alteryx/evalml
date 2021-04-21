
from skopt.space import Categorical

from evalml.pipelines import MulticlassClassificationPipeline


def test_multiclass_init():
    clf = MulticlassClassificationPipeline(component_graph=["Imputer", "One Hot Encoder", "Random Forest Classifier"])
    assert clf.parameters == {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean',
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'features_to_encode': None,
            'categories': None,
            'drop': 'if_binary',
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Random Forest Classifier': {
            'n_estimators': 100,
            'max_depth': 6,
            'n_jobs': -1
        }
    }
    assert clf.custom_hyperparameters is None
    assert clf.name == "Random Forest Classifier w/ Imputer + One Hot Encoder"
    assert clf.random_seed == 0
    custom_hyperparameters = {"Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", 'mean'])},
                              "Imputer_1": {"numeric_impute_strategy": Categorical(["median", 'mean'])},
                              "Random Forest Classifier": {"n_estimators": Categorical([50, 100])}}
    parameters = {
        "One Hot Encoder": {
            "top_n": 20
        }
    }
    clf = MulticlassClassificationPipeline(component_graph=["Imputer", "One Hot Encoder", "Random Forest Classifier"],
                                           parameters=parameters,
                                           custom_hyperparameters=custom_hyperparameters,
                                           custom_name="Custom Pipeline",
                                           random_seed=42)

    assert clf.parameters == {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean',
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 20,
            'features_to_encode': None,
            'categories': None,
            'drop': 'if_binary',
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Random Forest Classifier': {
            'n_estimators': 100,
            'max_depth': 6,
            'n_jobs': -1
        }
    }
    assert clf.custom_hyperparameters == custom_hyperparameters
    assert clf.name == "Custom Pipeline"
    assert clf.random_seed == 42
