<p align="center">
<img width=50% src="https://github.com/featurelabs/evalml/blob/master/docs/source/images/evalml_logo.png?raw=true" alt="Featuretools" />
</p>

[![CircleCI](https://circleci.com/gh/FeatureLabs/evalml.svg?style=svg&circle-token=9e0ce5e5f2db05f96fe92238fcde6d13963188b6)](https://circleci.com/gh/FeatureLabs/evalml)
[![codecov](https://codecov.io/gh/featurelabs/evalml/branch/master/graph/badge.svg?token=JDc0Ib7kYL)](https://codecov.io/gh/featurelabs/evalml)

EvalML is an AutoML library to build optimized machine learning pipelines for domain-specific objective functions.

**Key Functionality**

* **Domain-specific** - Includes repository of domain-specific objective functions and interface to define your own
* **End-to-end** - Constructs and optimizes pipelines that include imputation, feature selection, and a variety of modeling techniques
* **Guardrails** - Carefully cross-validates to prevent overfitting and warns you if training and testing results diverge

## Install
```shell
pip install evalml --index-url https://install.featurelabs.com/<KEY>
```

## Quick Start

#### Define objective
```python
from evalml import AutoClassificationSearch
from evalml.objectives import FraudCost


fraud_objective = FraudCost(
    retry_percentage=.5,
    interchange_fee=.02,
    fraud_payout_percentage=.75,
    amount_col="amount"
)
```

#### Run automl
```python
clf = AutoClassificationSearch(objective=fraud_objective,
                     max_pipelines=3)

clf.fit(X_train, y_train)
```

#### See all pipeline ranks
```python
clf.rankings
```

#### Get best pipeline and predict on new data

```python
pipeline = clf.best_pipeline
pipeline.predict(X_test)
```

## Next Steps

* [Configuring AutoClassifer]()
* [Defining your own objective functions]()
* [API Reference]()

Read more about EvalML in our [Documentation](evalml.featurelabs.com).

## Built at Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>
