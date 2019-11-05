.. _changelog:

Changelog
---------
**Future Releases**
    * Enhancements
        * Added ROC charts for binary classification :pr:`169`
        * Added support for unlimited pipelines with a max_time limit :pr:`70`
    * Fixes
    * Changes
        * Refactoring pipelines :pr:`108`
    * Documentation Changes
    * Testing Changes

**v0.5.0 Oct. 29, 2019**
    * Enhancements
        * Added basic one hot encoding :pr:`73`
        * Use enums for model_type :pr:`110`
        * Support for splitting regression datasets :pr:`112`
        * Auto-infer multiclass classification :pr:`99`
        * Added support for other units in max_time :pr:`125`
        * Detect highly null columns :pr:`121`
        * Added additional regression objectives :pr:`100`
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
