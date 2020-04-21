=============
API Reference
=============

.. currentmodule:: evalml.demos

Demo Datasets
=============

.. autosummary::
    :toctree: generated
    :nosignatures:

    load_fraud
    load_wine
    load_breast_cancer
    load_diabetes


.. currentmodule:: evalml.preprocessing

Preprocessing
=============

.. autosummary::
    :toctree: generated
    :nosignatures:

    load_data
    split_data


.. currentmodule:: evalml

AutoML
======

.. autosummary::
    :toctree: generated
    :template: class_with_properties.rst
    :nosignatures:

    AutoClassificationSearch
    AutoRegressionSearch


.. currentmodule:: evalml.model_family

Model Family
============

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ModelFamily


.. currentmodule:: evalml.pipelines.components

Components
==========

Transformers
~~~~~~~~~~~~

Encoders
--------

Encoders convert categorical or non-numerical features into numerical features.

.. autosummary::
    :toctree: generated
    :template: transformer_class.rst
    :nosignatures:

    OneHotEncoder

Imputers
--------

Imputers fill in missing data.

.. autosummary::
    :toctree: generated
    :template: transformer_class.rst
    :nosignatures:

    SimpleImputer


Scalers
-------

Scalers transform and standardize the range of data.

.. autosummary::
    :toctree: generated
    :template: transformer_class.rst
    :nosignatures:

    StandardScaler


Feature Selectors
-----------------

Feature selectors select a subset of relevant features for the model.

.. autosummary::
    :toctree: generated
    :template: transformer_class.rst
    :nosignatures:

    RFRegressorSelectFromModel
    RFClassifierSelectFromModel


Estimators
~~~~~~~~~~

Classifiers
-----------

Classifiers are models which can be trained to predict a class label from input data.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    LogisticRegressionClassifier
    RandomForestClassifier
    XGBoostClassifier


Regressors
-----------

Regressors are models which can be trained to predict a target value from input data.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    LinearRegressor
    RandomForestRegressor


.. currentmodule:: evalml.pipelines

Pipelines
=========

Pipeline Base Classes
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    PipelineBase
    BinaryClassificationPipeline
    MulticlassClassificationPipeline
    RegressionPipeline

Classification Pipelines
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: pipeline_class.rst
    :nosignatures:

    CatBoostBinaryClassificationPipeline
    CatBoostMulticlassClassificationPipeline
    LogisticRegressionBinaryPipeline
    LogisticRegressionMulticlassPipeline
    RFBinaryClassificationPipeline
    RFMulticlassClassificationPipeline
    XGBoostBinaryPipeline
    XGBoostMulticlassPipeline

Regression Pipelines
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: pipeline_class.rst
    :nosignatures:

    RFRegressionPipeline
    CatBoostRegressionPipeline
    LinearRegressionPipeline
    XGBoostRegressionPipeline


Pipeline Utils
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_pipelines
    list_model_families


.. currentmodule:: evalml.objectives

Objective Functions
====================

Domain-Specific Objectives
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    FraudCost
    LeadScoring


Classification Objectives
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    AccuracyBinary
    AccuracyMulticlass
    AUC
    AUCMacro
    AUCMicro
    AUCWeighted
    BalancedAccuracyBinary
    BalancedAccuracyMulticlass
    F1
    F1Micro
    F1Macro
    F1Weighted
    LogLossBinary
    LogLossMulticlass
    MCCBinary
    MCCMulticlass
    Precision
    PrecisionMicro
    PrecisionMacro
    PrecisionWeighted
    Recall
    RecallMicro
    RecallMacro
    RecallWeighted


Regression Objectives
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    R2
    MAE
    MSE
    MSLE
    MedianAE
    MaxError
    ExpVariance


Plot Metrics
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ROC
    ConfusionMatrix


.. currentmodule:: evalml.problem_types


Problem Types
=============

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ProblemTypes

.. autosummary::
    :toctree: generated
    :nosignatures:

    handle_problem_types


.. currentmodule:: evalml.tuners

Tuners
======

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    Tuner
    SKOptTuner
    GridSearchTuner
    RandomSearchTuner


.. currentmodule:: evalml.guardrails

Guardrails
=============

.. autosummary::
    :toctree: generated
    :nosignatures:

    detect_highly_null
    detect_label_leakage
    detect_outliers
    detect_id_columns


.. currentmodule:: evalml.utils

Utils
=====

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_to_seconds
    normalize_confusion_matrix
