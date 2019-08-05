# EvalML

EvalML is an AutoML library that optimizes models for domain-specific evaluation metrics.

## Install
```shell
pip install evalml
```

---

## API Reference

`evalml`

- `LeadScoring` - finds the optimal threshold for a label.
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
    
    - `predict` - predicts using the optimized threshold
        ```
        predict(y_prob)
        ```
        - `y_prob (DataFrame)` - probability estimates for each label

        Returns:
        - `DataFrame` - predictions using the optimized threshold
    - `score` - the cost function of threshold-based predictions
        ```
        score(y, y_prob)
        ```
        - `y (DataFrame)` - true labels
        - `y_prob (DataFrame)` - probability estimates for each label

        Returns:
        - `float` - weighted loss metric

---

## Built at Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>
