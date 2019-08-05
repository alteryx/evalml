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
    - `fit` - optimize threshold on probability estimates of the label
        ```
        fit(y_prob, y)
        ```
        - `y_prob (DataFrame)` - probability estimates of each label in train test
        - `y (DataFrame)` - true labels in train test

        Returns
        - `self` - instance of `LeadScoring`
    
    - `predict` - predicts using the optimized threshold
        ```
        predict(y_prob)
        ```
        - `y_prob (DataFrame)` - probability estimates for each label

        Returns
        - `Series` - estimated labels using the optimized threshold
    - `score` - the cost function on threshold-based predictions
        ```
        score(y, y_prob)
        ```
        - `y (DataFrame)` - true labels
        - `y_prob (DataFrame)` - probability estimates for each label

        Returns
        - `float` - weighted loss metric

`evalml.models`

- `Classifier` - machine learning model for classification-based prediction problems
    ```
    Classifier(random_state=0, options=None, cost_function=None, cost_function_holdout=.2, **kwargs)
    ```
    - `random_state (int)` - seed for the random number generator
    - `options (dict)` - extra parameters for unders sampling and resloving missing values
    - `cost_function (Metric)` - domin-specific metric to optimize model on
    - `cost_function_holdout (float)` - percent to holdout from train set to optimize cost function

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
        - `X (DataFrame)` - features in train set
        - `y (Series)` - true labels in train set
        - `options (dict)` - extra parameters for cross validation such as `folds`
        - `verbose (bool)` - whether to print during model training

        Returns
        - `DataFrame` - estimated metrics of model performance
    - `feature_importances` - return top k features. Model must be trained for feature selection.
        ```
        feature_importances(top_k=None, return_plot=True)
        ```
        - `top_k (int)` - number of top features to return
        - `return_plot (bool)` - whether to plot top k features

        Returns
        - `DataFrame` or `Plot` - top k features
    - `fit` - train model and select top features
        ```
        fit(X, y, verbose=True)
        ```
        - `X (DataFrame)` - features in train set
        - `y (Series)` - true labels in train set
        - `verbose (bool)` - whether to print during model training

        Returns
        - `self` - instance of `Classifier`
    - `predict_proba` - returns probability estimates for each label
        ```
        predict_proba(X)
        ```
        - `X (DataFrame)` - features

        Returns
        - `DataFrame` - probability estimates for each label
    - `predict` - predicts using the top selected features
        ```
        predict(X)
        ```
        - `X (DataFrame)` - features

        Returns
        - `Series` - estimated labels using the top selected features
    - `explain` - print model performance
        ```
        explain()
        ```
        Returns
        - `str` - performance report



## Built at Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>
