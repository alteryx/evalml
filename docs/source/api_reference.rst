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
    load_churn


.. currentmodule:: evalml.preprocessing

Preprocessing
=============

Utilities to preprocess data before using evalml.

.. autosummary::
    :toctree: generated
    :nosignatures:

    load_data
    drop_nan_target_rows
    target_distribution
    number_of_features
    split_data

Data Splitter Classes
~~~~~~~~~~~~~~~~~~~~~~
Data splitter classes for imbalanced classification datasets.

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    BalancedClassificationDataTVSplit
    BalancedClassificationDataCVSplit
    KMeansSMOTECVSplit
    KMeansSMOTETVSplit
    SMOTETomekCVSplit
    SMOTETomekTVSplit
    RandomUnderSamplerCVSplit
    RandomUnderSamplerTVSplit
    SMOTENCCVSplit
    SMOTENCTVSplit


.. currentmodule:: evalml.exceptions

Exceptions
=============

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    MethodPropertyNotFoundError
    PipelineNotFoundError
    ObjectiveNotFoundError
    IllFormattedClassNameError
    MissingComponentError
    ComponentNotYetFittedError
    PipelineNotYetFittedError
    AutoMLSearchException
    EnsembleMissingPipelinesError
    PipelineScoreError
    DataCheckInitError
    NullsInColumnWarning


.. currentmodule:: evalml.automl

AutoML
======

AutoML Search Classes
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: class_with_properties.rst
    :nosignatures:

    AutoMLSearch


AutoML Utils
~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_default_primary_search_objective
    make_data_splitter

.. currentmodule:: evalml.automl.automl_algorithm

AutoML Algorithm Classes
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: class_with_properties.rst
    :nosignatures:

    AutoMLAlgorithm
    IterativeAlgorithm


.. currentmodule:: evalml.automl.callbacks

AutoML Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    silent_error_callback
    log_error_callback
    raise_error_callback
    log_and_save_error_callback
    raise_and_save_error_callback


.. currentmodule:: evalml.pipelines

Pipelines
=========

Pipeline Base Classes
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: pipeline_base_class.rst
    :nosignatures:

    PipelineBase
    ClassificationPipeline
    BinaryClassificationPipeline
    MulticlassClassificationPipeline
    RegressionPipeline
    TimeSeriesClassificationPipeline
    TimeSeriesBinaryClassificationPipeline
    TimeSeriesMulticlassClassificationPipeline
    TimeSeriesRegressionPipeline

Classification Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: pipeline_class.rst
    :nosignatures:

    BaselineBinaryPipeline
    BaselineMulticlassPipeline
    ModeBaselineBinaryPipeline
    ModeBaselineMulticlassPipeline

Regression Pipelines
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: pipeline_class.rst
    :nosignatures:

    BaselineRegressionPipeline
    MeanBaselineRegressionPipeline
    TimeSeriesBaselineRegressionPipeline


.. currentmodule:: evalml.pipelines.utils

Pipeline Utils
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    make_pipeline
    make_pipeline_from_components
    generate_pipeline_code


.. currentmodule:: evalml.pipelines.components

Components
==========

Component Base Classes
~~~~~~~~~~~~~~~~~~~~~~
Components represent a step in a pipeline.

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ComponentBase
    Transformer
    Estimator

.. currentmodule:: evalml.pipelines.components.utils

Component Utils
~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    allowed_model_families
    get_estimators
    generate_component_code


.. currentmodule:: evalml.pipelines.components

Transformers
~~~~~~~~~~~~
Transformers are components that take in data as input and output transformed data.

.. autosummary::
    :toctree: generated
    :template: transformer_class.rst
    :nosignatures:

    DropColumns
    SelectColumns
    OneHotEncoder
    TargetEncoder
    PerColumnImputer
    Imputer
    SimpleImputer
    StandardScaler
    RFRegressorSelectFromModel
    RFClassifierSelectFromModel
    DropNullColumns
    DateTimeFeaturizer
    TextFeaturizer
    DelayedFeatureTransformer
    DFSTransformer
    PolynomialDetrender

Estimators
~~~~~~~~~~

Classifiers
-----------

Classifiers are components that output a predicted class label.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    CatBoostClassifier
    ElasticNetClassifier
    ExtraTreesClassifier
    RandomForestClassifier
    LightGBMClassifier
    LogisticRegressionClassifier
    XGBoostClassifier
    BaselineClassifier
    StackedEnsembleClassifier
    DecisionTreeClassifier
    KNeighborsClassifier
    SVMClassifier

Regressors
-----------

Regressors are components that output a predicted target value.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    ARIMARegressor
    CatBoostRegressor
    ElasticNetRegressor
    LinearRegressor
    ExtraTreesRegressor
    RandomForestRegressor
    XGBoostRegressor
    BaselineRegressor
    TimeSeriesBaselineEstimator
    StackedEnsembleRegressor
    DecisionTreeRegressor
    LightGBMRegressor
    SVMRegressor

.. currentmodule:: evalml.model_understanding

Model Understanding
===================

Utility Methods
~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    confusion_matrix
    normalize_confusion_matrix
    precision_recall_curve
    roc_curve
    calculate_permutation_importance
    binary_objective_vs_threshold
    get_prediction_vs_actual_over_time_data
    partial_dependence
    get_prediction_vs_actual_data
    get_linear_coefficients
    t_sne


Graph Utility Methods
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:


    graph_precision_recall_curve
    graph_roc_curve
    graph_confusion_matrix
    graph_permutation_importance
    graph_binary_objective_vs_threshold
    graph_prediction_vs_actual
    graph_prediction_vs_actual_over_time
    graph_partial_dependence
    graph_t_sne


.. currentmodule:: evalml.model_understanding.prediction_explanations

Prediction Explanations
~~~~~~~~~~~~~~~~~~~~~~~


.. autosummary::
    :toctree: generated
    :nosignatures:

    explain_predictions
    explain_predictions_best_worst


.. currentmodule:: evalml.objectives

Objective Functions
====================

Objective Base Classes
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: objective_base_class.rst
    :nosignatures:

    ObjectiveBase
    BinaryClassificationObjective
    MulticlassClassificationObjective
    RegressionObjective


Domain-Specific Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: objective_class.rst
    :nosignatures:

    FraudCost
    LeadScoring
    CostBenefitMatrix


Classification Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: objective_class.rst
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
    :template: objective_class.rst
    :nosignatures:

    R2
    MAE
    MAPE
    MSE
    MeanSquaredLogError
    MedianAE
    MaxError
    ExpVariance
    RootMeanSquaredError
    RootMeanSquaredLogError


Objective Utils
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_all_objective_names
    get_core_objectives
    get_core_objective_names
    get_non_core_objectives
    get_objective


.. currentmodule:: evalml.problem_types

Problem Types
=============

.. autosummary::
    :toctree: generated
    :nosignatures:

    handle_problem_types
    detect_problem_type

    :template: enum_class.rst

    ProblemTypes



.. currentmodule:: evalml.model_family

Model Family
============

.. autosummary::
    :toctree: generated
    :nosignatures:

    handle_model_family

    :template: enum_class.rst

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

Data Check Classes
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: data_check_class.rst
    :nosignatures:

    DataCheck
    InvalidTargetDataCheck
    HighlyNullDataCheck
    IDColumnsDataCheck
    TargetLeakageDataCheck
    OutliersDataCheck
    NoVarianceDataCheck
    ClassImbalanceDataCheck
    MulticollinearityDataCheck


.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    DataChecks
    DefaultDataChecks


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
    :template: enum_class.rst
    :nosignatures:

    DataCheckMessageType

Data Check Message Codes
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: enum_class.rst
    :nosignatures:

    DataCheckMessageCode


.. currentmodule:: evalml.utils

Utils
=====

General Utils
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    import_or_raise
    convert_to_seconds
    get_random_state
    get_random_seed
    pad_with_nans
    drop_rows_with_nans
    infer_feature_types
    save_plot
    is_all_numeric
    get_importable_subclasses
