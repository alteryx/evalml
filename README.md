<p align="center">
<img width=50% src="https://evalml-web-images.s3.amazonaws.com/evalml_horizontal.svg" alt="Featuretools" />
</p>

[![CircleCI](https://circleci.com/gh/alteryx/evalml/tree/main.svg?style=svg&circle-token=9e0ce5e5f2db05f96fe92238fcde6d13963188b6)](https://circleci.com/gh/alteryx/evalml/tree/main)
[![codecov](https://codecov.io/gh/alteryx/evalml/branch/main/graph/badge.svg?token=JDc0Ib7kYL)](https://codecov.io/gh/alteryx/evalml)
[![PyPI version](https://badge.fury.io/py/evalml.svg?maxAge=2592000)](https://badge.fury.io/py/evalml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/evalml.svg)](https://pypistats.org/packages/evalml)

EvalML is an AutoML library which builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.

**Key Functionality**

* **Automation** - Makes machine learning easier. Avoid training and tuning models by hand. Includes data quality checks, cross-validation and more.
* **Data Checks** - Catches and warns of problems with your data and problem setup before modeling.
* **End-to-end** - Constructs and optimizes pipelines that include state-of-the-art preprocessing, feature engineering, feature selection, and a variety of modeling techniques.
* **Model Understanding** - Provides tools to understand and introspect on models, to learn how they'll behave in your problem domain.
* **Domain-specific** - Includes repository of domain-specific objective functions and an interface to define your own.

## Install [from PyPI](https://pypi.org/project/evalml/)
```shell
pip install evalml
```

## Start

#### Load and split example data 
```python
import evalml
X, y = evalml.demos.load_breast_cancer()
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y)
```

#### Run AutoML
```python
from evalml.automl import AutoMLSearch
automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')
automl.search()
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

Read more about EvalML on our [documentation page](https://evalml.alteryx.com/):

* [Installation](https://evalml.alteryx.com/en/stable/install.html) and [getting started](https://evalml.alteryx.com/en/stable/start.html).
* [Tutorials](https://evalml.alteryx.com/en/stable/tutorials.html) on how to use EvalML.
* [User guide](https://evalml.alteryx.com/en/stable/user_guide.html) which describes EvalML's features.
* Full [API reference](https://evalml.alteryx.com/en/stable/api_reference.html)

## Built at Alteryx Innovation Labs
<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
