.. _changelog:

Changelog
---------
**Future Releases**
    * Enhancements
        * Added emacs buffers to .gitignore :pr:`350`
        * Add CatBoost (gradient-boosted trees) classification and regression components and pipelines :pr:`247`
        * Added Tuner abstract base class :pr:`351`
    * Fixes
        * Fixed ROC and confusion matrix plots not being calculated if user passed own additional_objectives :pr:`276`
    * Changes
        * Added n_estimators as a tunable parameter for XGBoost :pr:`307`
        * Remove unused parameter ObjectiveBase.fit_needs_proba :pr:`320`
        * Remove extraneous parameter component_type from all components :pr:`361`
        * Remove unused rankings.csv file :pr:`397`
        * Changed plot.feature_importance to show only non-zero feature importances by default, added optional parameter to show all :pr:`413`
    * Documentation Changes
        * Update release.md with instructions to release to internal license key :pr:`354`
    * Testing Changes
        * Added tests for utils (and moved current utils to gen_utils) :pr:`297`
        * Moved XGBoost install into it's own separate step on Windows using Conda :pr:`313`
        * Rewind pandas version to before 1.0.0, to diagnose test failures for that version :pr:`325`
        * Added dependency update checkin test :pr:`324`
        * Rewind XGBoost version to before 1.0.0 to diagnose test failures for that version :pr:`402`
        * Updated dependency jmespath :pr:`415`

**v0.6.0 Dec. 16, 2019**
    * Enhancements
        * Added ability to create a plot of feature importances :pr:`133`
        * Add early stopping to AutoML using patience and tolerance parameters :pr:`241`
        * Added ROC and confusion matrix metrics and plot for classification problems and introduce PipelineSearchPlots class :pr:`242`
        * Enhanced AutoML results with search order :pr:`260`
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
    * ``AutoClassificationSearch.results`` and ``AutoRegressionSearch.results`` now is a dictionary with ``pipeline_results`` and ``search_order`` keys. ``pipeline_results`` can be used to access a dictionary that is identical to the old ``.results`` dictionary. Whereas,``search_order`` returns a list of the search order in terms of pipeline id.
    * Pipelines now require an estimator as the last component in `component_list`. Slicing pipelines now throws an NotImplementedError to avoid returning Pipelines without an estimator.

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
