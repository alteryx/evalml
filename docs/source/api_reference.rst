=============
API Reference
=============

.. currentmodule:: evalml.demos

Demo Datasets
=============

.. autosummary::
    :toctree: generated
    :template: class.rst
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
    :template: class.rst
    :nosignatures:

    load_data
    split_data
    detect_label_leakage


.. currentmodule:: evalml

Models
======

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    AutoClassifier
    AutoRegressor

Model Types
===========

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    list_model_types

.. currentmodule:: evalml.pipelines

Pipelines
=========

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    get_pipelines
    save_pipeline
    load_pipeline
    RFClassificationPipeline
    XGBoostPipeline
    LogisticRegressionPipeline
    RFRegressionPipeline


Objective Functions
====================

.. currentmodule:: evalml.objectives


Domain Specific
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    FraudCost
    LeadScoring


Classification
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    F1
    F1_micro
    F1_macro
    F1_weighted
    Precision
    Precision_micro
    Precision_macro
    Precision_weighted
    Recall
    Recall_micro
    Recall_macro
    Recall_weighted
    AUC
    AUC_micro
    AUC_macro
    AUC_weighted
    LogLoss
    MCC

Regression
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    R2



.. currentmodule:: evalml.tuners

Tuners
======

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    SKOptTuner



