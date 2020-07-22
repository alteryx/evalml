<p align="center">
<img width=50% src="https://github.com/featurelabs/evalml/blob/main/docs/source/images/evalml_horizontal.svg?raw=true" alt="Featuretools" />
</p>

[![CircleCI](https://circleci.com/gh/FeatureLabs/evalml.svg?style=svg&circle-token=9e0ce5e5f2db05f96fe92238fcde6d13963188b6)](https://circleci.com/gh/FeatureLabs/evalml)
[![codecov](https://codecov.io/gh/featurelabs/evalml/branch/main/graph/badge.svg?token=JDc0Ib7kYL)](https://codecov.io/gh/featurelabs/evalml)

EvalML is an AutoML library that builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.

**Key Functionality**

* **Domain-specific** - Includes repository of domain-specific objective functions and interface to define your own.
* **End-to-end** - Constructs and optimizes pipelines that include state-of-the-art preprocessing, feature engineering, feature selection, and a variety of modeling techniques.
* **Automation** - Cross-validates to prevent overfitting and warns you if validation results diverge.
* **Data Checks** - Catches and warns of problems with your data and problem setup before modeling.

## Install [from PyPI](https://pypi.org/project/evalml/)
```shell
pip install evalml
```

## Start

#### Run AutoML
```python
from evalml.automl import AutoMLSearch
automl = AutoMLSearch(problem_type='binary')
automl.search(X_train, y_train)
```

#### View pipeline rankings
```python
automl.rankings
```

#### Get best pipeline and predict on new data
```python
pipeline = automl.best_pipeline
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

## Next Steps

Read more about EvalML in our [documentation](https://evalml.alteryx.com/):

* [Installation](https://evalml.alteryx.com/en/stable/install.html) and [getting started](https://evalml.alteryx.com/en/stable/start.html).
* [Tutorials](https://evalml.alteryx.com/en/stable/tutorials.html) on how to use EvalML.
* [User guide](https://evalml.alteryx.com/en/stable/user_guide.html) which describes EvalML's features.
* Full [API reference](https://evalml.alteryx.com/en/stable/api_reference.html)

## Built at Alteryx Innovation Labs
<a href="https://www.alteryx.com/innovation-labs">
    <img src="docs/source/images/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
