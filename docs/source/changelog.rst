.. _changelog:

Changelog
---------
**Future Releases**
    * Enhancements
        * Port over highly-null guardrail as a data check and define `DefaultDataChecks` and `DisableDataChecks` classes :pr:`745`
    * Fixes
    * Changes
        * Cleanup pipeline `score` code, and cleanup codecov :pr:`711`
        * Remove `pass` for abstract methods for codecov :pr:`730`
        * Added __str__ for AutoSearch object :pr:`675`
        * Add util methods to graph ROC and confusion matrix :pr:`720`
        * Refactor `AutoBase` to `AutoSearchBase` :pr:`758`
    * Documentation Changes
        * Add instructions to freeze `master` on `release.md` :pr:`726`
        * Update release instructions with more details :pr:`727` :pr:`733`
        * Add objective base classes to API reference :pr:`736`
        * Fix components API to match other modules :pr:`747`
    * Testing Changes
        * Delete codecov yml, use codecov.io's default :pr:`732`
        * Added unit tests for fraud cost, lead scoring, and standard metric objectives, :pr:`741`

.. warning::

    **Breaking Changes**

    * Moved ROC and confusion matrix methods from `evalml.pipeline.plot_utils` to `evalml.pipeline.graph_utils` :pr:`720`



**v0.9.0 Apr. 27, 2020**
    * Enhancements
        * Added accuracy as an standard objective :pr:`624`
        * Added verbose parameter to load_fraud :pr:`560`
        * Added Balanced Accuracy metric for binary, multiclass :pr:`612` :pr:`661`
        * Added XGBoost regressor and XGBoost regression pipeline :pr:`666`
        * Added Accuracy metric for multiclass :pr:`672`
        * Added objective name in `AutoBase.describe_pipeline` :pr:`686`
        * Added `DataCheck` and `DataChecks`, `Message` classes and relevant subclasses :pr:`739`
    * Fixes
        * Removed direct access to `cls.component_graph` :pr:`595`
        * Add testing files to .gitignore :pr:`625`
        * Remove circular dependencies from `Makefile` :pr:`637`
        * Add error case for `normalize_confusion_matrix()` :pr:`640`
        * Fixed XGBoostClassifier and XGBoostRegressor bug with feature names that contain [, ], or < :pr:`659`
        * Update make_pipeline_graph to not accidentally create empty file when testing if path is valid :pr:`649`
        * Fix pip installation warning about docsutils version, from boto dependency :pr:`664`
        * Removed zero division warning for F1/precision/recall metrics :pr:`671`
        * Fixed `summary` for pipelines without estimators :pr:`707`
    * Changes
        * Updated default objective for binary/multiseries classification to log loss :pr:`613`
        * Created classification and regression pipeline subclasses and removed objective as an attribute of pipeline classes :pr:`405`
        * Changed the output of `score` to return one dictionary :pr:`429`
        * Created binary and multiclass objective subclasses :pr:`504`
        * Updated objectives API :pr:`445`
        * Removed call to `get_plot_data` from AutoML :pr:`615`
        * Set `raise_error` to default to True for AutoML classes :pr:`638`
        * Remove unnecessary "u" prefixes on some unicode strings :pr:`641`
        * Changed one-hot encoder to return uint8 dtypes instead of ints :pr:`653`
        * Pipeline `_name` field changed to `custom_name` :pr:`650`
        * Removed `graphs.py` and moved methods into `PipelineBase` :pr:`657`, :pr:`665`
        * Remove s3fs as a dev dependency :pr:`664`
        * Changed requirements-parser to be a core dependency :pr:`673`
        * Replace `supported_problem_types` field on pipelines with `problem_type` attribute on base classes :pr:`678`
        * Changed AutoML to only show best results for a given pipeline template in `rankings`, added `full_rankings` property to show all :pr:`682`
        * Update `ModelFamily` values: don't list xgboost/catboost as classifiers now that we have regression pipelines for them :pr:`677`
        * Changed AutoML's `describe_pipeline` to get problem type from pipeline instead :pr:`685`
        * Standardize `import_or_raise` error messages :pr:`683`
        * Updated argument order of objectives to align with sklearn's :pr:`698`
        * Renamed `pipeline.feature_importance_graph` to `pipeline.graph_feature_importances` :pr:`700`
        * Moved ROC and confusion matrix methods to `evalml.pipelines.plot_utils` :pr:`704`
        * Renamed `MultiClassificationObjective` to `MulticlassClassificationObjective`, to align with pipeline naming scheme :pr:`715`
    * Documentation Changes
        * Fixed some sphinx warnings :pr:`593`
        * Fixed docstring for AutoClassificationSearch with correct command :pr:`599`
        * Limit readthedocs formats to pdf, not htmlzip and epub :pr:`594` :pr:`600`
        * Clean up objectives API documentation :pr:`605`
        * Fixed function on Exploring search results page :pr:`604`
        * Update release process doc :pr:`567`
        * AutoClassificationSearch and AutoRegressionSearch show inherited methods in API reference :pr:`651`
        * Fixed improperly formatted code in breaking changes for changelog :pr:`655`
        * Added configuration to treat Sphinx warnings as errors :pr:`660`
        * Removed separate plotting section for pipelines in API reference :pr:`657`, :pr:`665`
        * Have leads example notebook load S3 files using https, so we can delete s3fs dev dependency :pr:`664`
        * Categorized components in API reference and added descriptions for each category :pr:`663`
        * Fixed Sphinx warnings about BalancedAccuracy objective :pr:`669`
        * Updated API reference to include missing components and clean up pipeline docstrings :pr:`689`
        * Reorganize API ref, and clarify pipeline sub-titles :pr:`688`
        * Add and update preprocessing utils in API reference :pr:`687`
        * Added inheritance diagrams to API reference :pr:`695`
        * Documented which default objective AutoML optimizes for :pr:`699`
        * Create seperate install page :pr:`701`
        * Include more utils in API ref, like `import_or_raise` :pr:`704`
        * Add more color to pipeline documentation :pr:`705`
    * Testing Changes
        * Matched install commands of `check_latest_dependencies` test and it's GitHub action :pr:`578`
        * Added Github app to auto assign PR author as assignee :pr:`477`
        * Removed unneeded conda installation of xgboost in windows checkin tests :pr:`618`
        * Update graph tests to always use tmpfile dir :pr:`649`
        * Changelog checkin test workaround for release PRs: If 'future release' section is empty of PR refs, pass check :pr:`658`
        * Add changelog checkin test exception for `dep-update` branch :pr:`723`

.. warning::

    **Breaking Changes**

    * Pipelines will now no longer take an objective parameter during instantiation, and will no longer have an objective attribute.
    * ``fit()`` and ``predict()`` now use an optional ``objective`` parameter, which is only used in binary classification pipelines to fit for a specific objective.
    * ``score()`` will now use a required ``objectives`` parameter that is used to determine all the objectives to score on. This differs from the previous behavior, where the pipeline's objective was scored on regardless.
    * ``score()`` will now return one dictionary of all objective scores.
    * ``ROC`` and ``ConfusionMatrix`` plot methods via ``Auto(*).plot`` have been removed by :pr:`615` and are replaced by ``roc_curve`` and ``confusion_matrix`` in `evamlm.pipelines.plot_utils`` in :pr:`704`
    * ``normalize_confusion_matrix`` has been moved to ``evalml.pipelines.plot_utils`` :pr:`704`
    * Pipelines ``_name`` field changed to ``custom_name``
    * Pipelines ``supported_problem_types`` field is removed because it is no longer necessary :pr:`678`
    * Updated argument order of objectives' `objective_function` to align with sklearn :pr:`698`
    * `pipeline.feature_importance_graph` has been renamed to `pipeline.graph_feature_importances` in :pr:`700`
    * Removed unsupported ``MSLE`` objective :pr:`704`


**v0.8.0 Apr. 1, 2020**
    * Enhancements
        * Add normalization option and information to confusion matrix :pr:`484`
        * Add util function to drop rows with NaN values :pr:`487`
        * Renamed `PipelineBase.name` as `PipelineBase.summary` and redefined `PipelineBase.name` as class property :pr:`491`
        * Added access to parameters in Pipelines with `PipelineBase.parameters` (used to be return of `PipelineBase.describe`) :pr:`501`
        * Added `fill_value` parameter for SimpleImputer :pr:`509`
        * Added functionality to override component hyperparameters and made pipelines take hyperparemeters from components :pr:`516`
        * Allow numpy.random.RandomState for random_state parameters :pr:`556`
    * Fixes
        * Removed unused dependency `matplotlib`, and move `category_encoders` to test reqs :pr:`572`
    * Changes
        * Undo version cap in XGBoost placed in :pr:`402` and allowed all released of XGBoost :pr:`407`
        * Support pandas 1.0.0 :pr:`486`
        * Made all references to the logger static :pr:`503`
        * Refactored `model_type` parameter for components and pipelines to `model_family` :pr:`507`
        * Refactored `problem_types` for pipelines and components into `supported_problem_types` :pr:`515`
        * Moved `pipelines/utils.save_pipeline` and `pipelines/utils.load_pipeline` to `PipelineBase.save` and `PipelineBase.load` :pr:`526`
        * Limit number of categories encoded by OneHotEncoder :pr:`517`
    * Documentation Changes
        * Updated API reference to remove PipelinePlot and added moved PipelineBase plotting methods :pr:`483`
        * Add code style and github issue guides :pr:`463` :pr:`512`
        * Updated API reference for to surface class variables for pipelines and components :pr:`537`
        * Fixed README documentation link :pr:`535`
        * Unhid PR references in changelog :pr:`656`
    * Testing Changes
        * Added automated dependency check PR :pr:`482`, :pr:`505`
        * Updated automated dependency check comment :pr:`497`
        * Have build_docs job use python executor, so that env vars are set properly :pr:`547`
        * Added simple test to make sure OneHotEncoder's top_n works with large number of categories :pr:`552`
        * Run windows unit tests on PRs :pr:`557`


.. warning::

    **Breaking Changes**

    * ``AutoClassificationSearch`` and ``AutoRegressionSearch``'s ``model_types`` parameter has been refactored into ``allowed_model_families``
    * ``ModelTypes`` enum has been changed to ``ModelFamily``
    * Components and Pipelines now have a ``model_family`` field instead of ``model_type``
    * ``get_pipelines`` utility function now accepts ``model_families`` as an argument instead of ``model_types``
    * ``PipelineBase.name`` no longer returns structure of pipeline and has been replaced by ``PipelineBase.summary``
    * ``PipelineBase.problem_types`` and ``Estimator.problem_types`` has been renamed to ``supported_problem_types``
    * ``pipelines/utils.save_pipeline`` and ``pipelines/utils.load_pipeline`` moved to ``PipelineBase.save`` and ``PipelineBase.load``


**v0.7.0 Mar. 9, 2020**
    * Enhancements
        * Added emacs buffers to .gitignore :pr:`350`
        * Add CatBoost (gradient-boosted trees) classification and regression components and pipelines :pr:`247`
        * Added Tuner abstract base class :pr:`351`
        * Added n_jobs as parameter for AutoClassificationSearch and AutoRegressionSearch :pr:`403`
        * Changed colors of confusion matrix to shades of blue and updated axis order to match scikit-learn's :pr:`426`
        * Added PipelineBase graph and feature_importance_graph methods, moved from previous location :pr:`423`
        * Added support for python 3.8 :pr:`462`
    * Fixes
        * Fixed ROC and confusion matrix plots not being calculated if user passed own additional_objectives :pr:`276`
        * Fixed ReadtheDocs FileNotFoundError exception for fraud dataset :pr:`439`
    * Changes
        * Added n_estimators as a tunable parameter for XGBoost :pr:`307`
        * Remove unused parameter ObjectiveBase.fit_needs_proba :pr:`320`
        * Remove extraneous parameter component_type from all components :pr:`361`
        * Remove unused rankings.csv file :pr:`397`
        * Downloaded demo and test datasets so unit tests can run offline :pr:`408`
        * Remove `_needs_fitting` attribute from Components :pr:`398`
        * Changed plot.feature_importance to show only non-zero feature importances by default, added optional parameter to show all :pr:`413`
        * Refactored `PipelineBase` to take in parameter dictionary and moved pipeline metadata to class attribute :pr:`421`
        * Dropped support for Python 3.5 :pr:`438`
        * Removed unused `apply.py` file :pr:`449`
        * Clean up requirements.txt to remove unused deps :pr:`451`
        * Support installation without all required dependencies :pr:`459`
    * Documentation Changes
        * Update release.md with instructions to release to internal license key :pr:`354`
    * Testing Changes
        * Added tests for utils (and moved current utils to gen_utils) :pr:`297`
        * Moved XGBoost install into it's own separate step on Windows using Conda :pr:`313`
        * Rewind pandas version to before 1.0.0, to diagnose test failures for that version :pr:`325`
        * Added dependency update checkin test :pr:`324`
        * Rewind XGBoost version to before 1.0.0 to diagnose test failures for that version :pr:`402`
        * Update dependency check to use a whitelist :pr:`417`
        * Update unit test jobs to not install dev deps :pr:`455`

.. warning::

    **Breaking Changes**

    * Python 3.5 will not be actively supported.

**v0.6.0 Dec. 16, 2019**
    * Enhancements
        * Added ability to create a plot of feature importances :pr:`133`
        * Add early stopping to AutoML using patience and tolerance parameters :pr:`241`
        * Added ROC and confusion matrix metrics and plot for classification problems and introduce PipelineSearchPlots class :pr:`242`
        * Enhanced AutoML results with search order :pr:`260`
        * Added utility function to show system and environment information :pr:`300`
    * Fixes
        * Lower botocore requirement :pr:`235`
        * Fixed decision_function calculation for FraudCost objective :pr:`254`
        * Fixed return value of Recall metrics :pr:`264`
        * Components return `self` on fit :pr:`289`
    * Changes
        * Renamed automl classes to AutoRegressionSearch and AutoClassificationSearch :pr:`287`
        * Updating demo datasets to retain column names :pr:`223`
        * Moving pipeline visualization to PipelinePlots class :pr:`228`
        * Standarizing inputs as pd.Dataframe / pd.Series :pr:`130`
        * Enforcing that pipelines must have an estimator as last component :pr:`277`
        * Added ipywidgets as a dependency in requirements.txt :pr:`278`
        * Added Random and Grid Search Tuners :pr:`240`
    * Documentation Changes
        * Adding class properties to API reference :pr:`244`
        * Fix and filter FutureWarnings from scikit-learn :pr:`249`, :pr:`257`
        * Adding Linear Regression to API reference and cleaning up some Sphinx warnings :pr:`227`
    * Testing Changes
        * Added support for testing on Windows with CircleCI :pr:`226`
        * Added support for doctests :pr:`233`

.. warning::

    **Breaking Changes**

    * The ``fit()`` method for ``AutoClassifier`` and ``AutoRegressor`` has been renamed to ``search()``.
    * ``AutoClassifier`` has been renamed to ``AutoClassificationSearch``
    * ``AutoRegressor`` has been renamed to ``AutoRegressionSearch``
    * ``AutoClassificationSearch.results`` and ``AutoRegressionSearch.results`` now is a dictionary with ``pipeline_results`` and ``search_order`` keys. ``pipeline_results`` can be used to access a dictionary that is identical to the old ``.results`` dictionary. Whereas, ``search_order`` returns a list of the search order in terms of ``pipeline_id``.
    * Pipelines now require an estimator as the last component in ``component_list``. Slicing pipelines now throws an ``NotImplementedError`` to avoid returning pipelines without an estimator.

**v0.5.2 Nov. 18, 2019**
    * Enhancements
        * Adding basic pipeline structure visualization :pr:`211`
    * Documentation Changes
        * Added notebooks to build process :pr:`212`

**v0.5.1 Nov. 15, 2019**
    * Enhancements
        * Added basic outlier detection guardrail :pr:`151`
        * Added basic ID column guardrail :pr:`135`
        * Added support for unlimited pipelines with a max_time limit :pr:`70`
        * Updated .readthedocs.yaml to successfully build :pr:`188`
    * Fixes
        * Removed MSLE from default additional objectives :pr:`203`
        * Fixed random_state passed in pipelines :pr:`204`
        * Fixed slow down in RFRegressor :pr:`206`
    * Changes
        * Pulled information for describe_pipeline from pipeline's new describe method :pr:`190`
        * Refactored pipelines :pr:`108`
        * Removed guardrails from Auto(*) :pr:`202`, :pr:`208`
    * Documentation Changes
        * Updated documentation to show max_time enhancements :pr:`189`
        * Updated release instructions for RTD :pr:`193`
        * Added notebooks to build process :pr:`212`
        * Added contributing instructions :pr:`213`
        * Added new content :pr:`222`

**v0.5.0 Oct. 29, 2019**
    * Enhancements
        * Added basic one hot encoding :pr:`73`
        * Use enums for model_type :pr:`110`
        * Support for splitting regression datasets :pr:`112`
        * Auto-infer multiclass classification :pr:`99`
        * Added support for other units in max_time :pr:`125`
        * Detect highly null columns :pr:`121`
        * Added additional regression objectives :pr:`100`
        * Show an interactive iteration vs. score plot when using fit() :pr:`134`
    * Fixes
        * Reordered `describe_pipeline` :pr:`94`
        * Added type check for model_type :pr:`109`
        * Fixed `s` units when setting string max_time :pr:`132`
        * Fix objectives not appearing in API documentation :pr:`150`
    * Changes
        * Reorganized tests :pr:`93`
        * Moved logging to its own module :pr:`119`
        * Show progress bar history :pr:`111`
        * Using cloudpickle instead of pickle to allow unloading of custom objectives :pr:`113`
        * Removed render.py :pr:`154`
    * Documentation Changes
        * Update release instructions :pr:`140`
        * Include additional_objectives parameter :pr:`124`
        * Added Changelog :pr:`136`
    * Testing Changes
        * Code coverage :pr:`90`
        * Added CircleCI tests for other Python versions :pr:`104`
        * Added doc notebooks as tests :pr:`139`
        * Test metadata for CircleCI and 2 core parallelism :pr:`137`

**v0.4.1 Sep. 16, 2019**
    * Enhancements
        * Added AutoML for classification and regressor using Autobase and Skopt :pr:`7` :pr:`9`
        * Implemented standard classification and regression metrics :pr:`7`
        * Added logistic regression, random forest, and XGBoost pipelines :pr:`7`
        * Implemented support for custom objectives :pr:`15`
        * Feature importance for pipelines :pr:`18`
        * Serialization for pipelines :pr:`19`
        * Allow fitting on objectives for optimal threshold :pr:`27`
        * Added detect label leakage :pr:`31`
        * Implemented callbacks :pr:`42`
        * Allow for multiclass classification :pr:`21`
        * Added support for additional objectives :pr:`79`
    * Fixes
        * Fixed feature selection in pipelines :pr:`13`
        * Made random_seed usage consistent :pr:`45`
    * Documentation Changes
        * Documentation Changes
        * Added docstrings :pr:`6`
        * Created notebooks for docs :pr:`6`
        * Initialized readthedocs EvalML :pr:`6`
        * Added favicon :pr:`38`
    * Testing Changes
        * Added testing for loading data :pr:`39`

**v0.2.0 Aug. 13, 2019**
    * Enhancements
        * Created fraud detection objective :pr:`4`

**v0.1.0 July. 31, 2019**
    * *First Release*
    * Enhancements
        * Added lead scoring objecitve :pr:`1`
        * Added basic classifier :pr:`1`
    * Documentation Changes
        * Initialized Sphinx for docs :pr:`1`
