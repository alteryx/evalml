<p align="center">
<img width=50% src="https://evalml-web-images.s3.amazonaws.com/evalml_horizontal.svg" alt="Featuretools" />
</p>

[![GitHub Actions](https://github.com/alteryx/evalml/actions/workflows/linux_unit_tests.yml/badge.svg?branch=main)](https://github.com/alteryx/evalml/actions/workflows/linux_unit_tests.yml?query=branch%3Amain)
[![Codecov.io](https://codecov.io/gh/alteryx/evalml/branch/main/graph/badge.svg?token=JDc0Ib7kYL)](https://codecov.io/gh/alteryx/evalml)
[![PyPI](https://badge.fury.io/py/evalml.svg?maxAge=2592000)](https://badge.fury.io/py/evalml)
[![PyPI Stats](https://img.shields.io/pypi/dm/evalml.svg)](https://pypistats.org/packages/evalml)

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
### Add-ons

#### Time Series support with Facebook's Prophet

To support the `Prophet` time series estimator, be sure to install it as an extra requirement. Please note that this may take a few minutes.
Prophet is currently only supported via pip installation in EvalML.
```shell
pip install evalml[prophet]
```
Another option for installing Prophet with CmdStan as a backend is to use `make installdeps-prophet`.
This command will do the following:
- Pip install `cmdstanpy==0.9.68`
- Execute the `install_cmdstan.py` script found within your `site-packages/cmdstanpy` which builds `cmdstan` in your `site-packages`.
- Install `Prophet==1.0.1` with the `CMDSTAN` and `STAN_BACKEND` environment variables set.

If the `site-packages` path is incorrect or you'd like to specify a different one, just run `make installdeps-prophet SITE_PACKAGES_DIR="<path_to_your_site_packages>"`.

If you'd like to have more fine-tuned control over the installation steps for Prophet, such as specifying the backend, follow these steps:

For CmdStanPy as a backend:
1. `pip install cmdstanpy==0.9.68`
2. `python <path_to_installed_cmdstanpy>/install_cmdstan.py --dir <path_to_build_cmdstan>`
3. `CMDSTAN=<path_to_build_cmdstan>/cmdstan-2.27.0 STAN_BACKEND=CMDSTANPY pip install prophet==1.0.1`

For PyStan as a backend (PyStan is used by default):
1. `pip install prophet==1.0.1`

#### Update checker

Receive automatic notifications of new EvalML releases
```shell
pip install evalml[update_checker]
```

## Start

#### Load and split example data 
```python
import evalml
X, y = evalml.demos.load_breast_cancer()
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='binary')
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
