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

Utilities to preprocess data before using evalml.
 
.. autosummary::
    :toctree: generated
    :nosignatures:

    drop_nan_target_rows
    label_distribution
    load_data
    number_of_features
    split_data


.. currentmodule:: evalml.automl

AutoML
======

.. autosummary::
    :toctree: generated
    :template: class_with_properties.rst
    :nosignatures:

    AutoClassificationSearch
    AutoRegressionSearch


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
    ClassificationPipeline
    BinaryClassificationPipeline
    MulticlassClassificationPipeline
    RegressionPipeline

Classification Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~~~~~

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

    all_pipelines
    get_pipelines
    list_model_families


Pipeline Plot Utils
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    roc_curve
    confusion_matrix
    normalize_confusion_matrix


.. currentmodule:: evalml.pipelines.components

Components
==========

Component Base Classes
~~~~~~~~~~~~~~~~~~~~~~
Components make up a step of a pipeline and either transforms or predicts data.

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ComponentBase
    Transformer
    Estimator

Transformers
~~~~~~~~~~~~
Transformers are components that take in data as input and output transformed data.

.. autosummary::
    :toctree: generated
    :template: transformer_class.rst
    :nosignatures:

    OneHotEncoder
    SimpleImputer
    StandardScaler
    RFRegressorSelectFromModel
    RFClassifierSelectFromModel


Estimators
~~~~~~~~~~

Classifiers
-----------

Classifiers are components that take in data as input and output a predicted class label.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    CatBoostClassifier
    RandomForestClassifier
    LogisticRegressionClassifier
    XGBoostClassifier

Regressors
-----------

Regressors are components that take in data as input and output a predicted a target value that exists in a continuous range.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    CatBoostRegressor
    LinearRegressor
    RandomForestRegressor
    XGBoostRegressor


.. currentmodule:: evalml.objectives

Objective Functions
====================

Objective Base Classes
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ObjectiveBase
    BinaryClassificationObjective
    MulticlassClassificationObjective
    RegressionObjective



Domain-Specific Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    FraudCost
    LeadScoring


Classification Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    R2
    MAE
    MSE
    MedianAE
    MaxError
    ExpVariance


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


.. currentmodule:: evalml.model_family

Model Family
============

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ModelFamily


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


.. currentmodule:: evalml.data_checks

Data Checks
===========

Data Check Base Classes
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: data_check_class.rst
    :nosignatures:

    DataCheck


.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    DataChecks


Data Check Messages
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: data_check_message.rst
    :nosignatures:

    DataCheckMessage
    DataCheckError
    DataCheckWarning


Data Check Message Types
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    DataCheckMessageType


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

    import_or_raise
    convert_to_seconds
    get_random_state
    get_random_seed

