.. _changelog:

Changelog
---------
**Future Releases**
    * Enhancements
        * Added ROC and confusion matrix metrics and plot for classification problems and introduce PipelineSearchPlots class :pr:`242`
    * Fixes
        * Lower botocore requirement :pr:`235`
    * Changes
        * Updating demo datasets to retain column names :pr:`223`
        * Moving pipeline visualization to PipelinePlots class :pr:`228`
        * Standarizing inputs as pd.Dataframe / pd.Series :pr:`130`
    * Documentation Changes
        * Adding Linear Regression to API reference and cleaning up some Sphinx warnings :pr:`227`
    * Testing Changes
        * Added support for testing on Windows with CircleCI :pr:`226`
        * Added support for doctests :pr:`233`

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
