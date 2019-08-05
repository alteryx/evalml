# EvalML

EvalML is an AutoML library that optimizes models for domain-specific evaluation metrics.

## Install
```shell
pip install evalml
```

## API Reference

`evalml`

- `LeadScoring` - finds the optimal threshold on probability estimates of a label.
    ```
    LeadScoring(label=1, true_positives=1, false_positives=1, threshold=.5, verbose=True)
    ```
    - `label (int)` - label to optimize threshold for
    - `true_positives (int)` - value for each true positive
    - `false_positives (int)` - value for each false positive
    - `threshold (float)` - starting value for threshold
    - `verbose (bool)` - whether to print while optimizing threshold

    Methods
    - `fit` - fits data to optimize threshold
        ```
        fit(y_prob, y)
        ```
        - `y_prob (DataFrame)` - probability estimates for each label
        - `y (DataFrame)` - true labels

        Returns
        - `self` - instance of `LeadScoring`
    
    - `predict` - predicts using the optimized threshold
        ```
        predict(y_prob)
        ```
        - `y_prob (DataFrame)` - probability estimates for each label

        Returns
        - `DataFrame` - predictions using the optimized threshold
    - `score` - the cost function on threshold-based predictions
        ```
        score(y, y_prob)
        ```
        - `y (DataFrame)` - true labels
        - `y_prob (DataFrame)` - probability estimates for each label

        Returns
        - `float` - weighted loss metric

`evalml.models`

- `Classifier` - 
    ```
    Classifier(random_state=0, options=None, cost_function=None, cost_function_holdout=.2, **kwargs)
    ```
    - `` - 

    Methods
    - `score` - returns final metrics to evalute model performance. Also, can compare final performance with estimated performance.
        ```
        score(X, y)
        ```
        - `X (DataFrame)` - features in test set for model predictions
        - `y (Series)` - true labels in test set for model evaluation

        Returns
        - `DataFrame` - final metrics of model performance
    - `estimate_performance` - evaluates model performance by cross validating on train set
        ```
        estimate_performance(X, y, options=None, verbose=True)
        ```
        - `X (DataFrame)` - features in train set for model predictions
        - `y (Series)` - true labels in test set for model evaluation
        - `options (dict)` - extra parameters such as cross validation `folds`
        - `verbose (bool)` - whether to print during model training

        Returns
        - `DataFrame` - estimated metrics of model performance


## Built at Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>
