"""
Copyright (c) 2019 Feature Labs, Inc.

The usage of this software is governed by the Feature Labs End User License Agreement available at https://www.featurelabs.com/eula/. If you do not agree to the terms set out in this agreement, do not use the software, and immediately contact Feature Labs or your supplier.
"""
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from . import apply, render


class Classifier:
    def __init__(self, random_state=0, options=None, cost_function=None, cost_function_holdout=.2, **kwargs):
        self._label = None
        self._feature_importances = None
        self.selected_features = None
        self.scores_estimate = None
        self.scores = None
        self.cost_function_holdout = cost_function_holdout

        self.options = options or {}
        self.random_state = random_state
        self.model = XGBClassifier(
            random_state=self.random_state,
            **kwargs,
        )
        self.cost_function = cost_function

    def score(self, X, y):
        y_hat = self.predict(X)
        scores = apply.score(y, y_hat)

        if self.cost_function is not None:
            y_prob = self.predict_proba(X)
            name = self.cost_function.__class__.__name__
            scores[name] = self.cost_function.score(y, y_prob)

        self.scores = render.scores(scores)
        if self.scores_estimate is not None:
            self.scores = apply.compare_scores(
                self.scores,
                self.scores_estimate,
            )

            if self.cost_function is not None:
                name = self.cost_function.__class__.__name__
                self.scores.loc[name] = [self.scores.loc[name].values[0], 'N/A', 'N/A']

        return self.scores

    def estimate_performance(self, X, y, options=None, verbose=True):
        self.scores_estimate = None
        options = options or {}
        folds = options.get('folds', 2)
        cv = StratifiedKFold(random_state=self.random_state, n_splits=folds)

        fold, scores = 0, []
        for train, test in cv.split(X, y):
            if verbose:
                fold += 1
                info = '\n\n#{} Fold\n{}'
                print(info.format(fold, '-' * 10))

            x_train = X.loc[X.index[train]]
            y_train = y.loc[y.index[train]]
            self.fit(x_train, y_train, verbose=verbose)

            x_test = X.loc[X.index[test]]
            y_test = y.loc[y.index[test]]

            score = self.score(x_test, y_test)
            scores.append(score['Scores'].to_dict())

        self.scores_estimate = render.scores_estimate(scores)
        name = 'Stratified ({}-Fold CV)'.format(folds)
        return self.scores_estimate.rename_axis(name, axis=1)

    def feature_importances(self, top_k=None, return_plot=True):
        if self._feature_importances is not None:
            feature_importances = render.feature_importances(self._feature_importances)

            top_k = top_k or len(feature_importances)

            if not return_plot:
                return feature_importances.head(top_k)

            return render.feature_importances_plot(
                feature_importances.head(top_k),
                color='#4285F4',
                figsize=(8, 5),
            )

    def _under_sample(self, X, y, labels, verbose=True):
        min_class = labels.idxmin()

        if verbose:
            info = 'Under sampling minority class "{}".'
            print(info.format(min_class), end='\n')

        weights = labels.mul(0).add(1).to_dict()
        weights.update(self.options.get('weights', {}))

        X, y = apply.resample_labels(
            X,
            y,
            label=min_class,
            weights=weights,
            random_state=self.random_state,
        )

        labels = y.value_counts()
        return X, y, labels

    def _select_features(self, X, y, top_k, verbose=True):
        self.model.fit(X, y)
        self._feature_importances = dict(zip(X.columns, self.model.feature_importances_))

        if verbose:
            print('Selecting top {} features.'.format(top_k))

        self.selected_features = self.feature_importances(top_k=top_k, return_plot=False).index.tolist()

    def fit(self, X, y, verbose=True):
        self._label = y.name

        # resolve missing values
        max_na_percent = self.options.get('max_na_percent')

        if max_na_percent:
            len_columns = len(X.columns)
            thresh = int(len(X) * (1 - max_na_percent))
            X = X.dropna(thresh=thresh, axis=1)

            n = len_columns - len(X.columns)
            if verbose and n > 0:
                info = 'Dropping {} columns with more than {:.2f}% missing values.'
                print(info.format(n, max_na_percent * 100), end='\n')

        labels = y.value_counts()
        imbalanced = labels.nunique() > 1
        undersample = self.options.get('undersample', True)

        if self.cost_function is None:
            if imbalanced and undersample:
                X, y, labels = self._under_sample(X, y, labels, verbose=verbose)

            k = labels.loc[labels.idxmin()]
            self._select_features(X, y, top_k=k, verbose=verbose)

            if verbose:
                info = 'Training on {} examples.\n\n{}\n'
                n_examples = len(X[self.selected_features])
                distribution = render.label_distribution(labels.div(len(y)))
                print(info.format(n_examples, distribution))

            self.model.fit(X[self.selected_features], y)

        else:
            # split for training model and threhsold
            params = dict(holdout=self.cost_function_holdout, random_state=self.random_state)
            X_train, X_thresh, y_train, y_thresh = apply.split_data(X, y, **params)

            # under sample for feature selection
            if imbalanced and undersample:
                X, y, labels = self._under_sample(X, y, labels, verbose=verbose)

            k = labels.loc[labels.idxmin()]
            self._select_features(X, y, top_k=k, verbose=verbose)

            labels = y_train.value_counts()

            # under sample for model training
            if imbalanced and undersample:
                X_train, y_train, labels = self._under_sample(X_train, y_train, labels, verbose=verbose)

            if verbose:
                info = 'Training on {} examples.\n\n{}\n'
                n_examples = len(X_train[self.selected_features])
                distribution = render.label_distribution(labels.div(len(y_train)))
                print(info.format(n_examples, distribution))

            self.model.fit(X_train[self.selected_features], y_train)

            y_prob = self.predict_proba(X_thresh[self.selected_features])
            self.cost_function.fit(y_prob, y_thresh)

        return self

    def predict_proba(self, X):
        if self.selected_features is not None:
            X = X[self.selected_features]

        y_prob = self.model.predict_proba(X)
        df = DataFrame(y_prob, index=X.index, columns=self.model.classes_)
        df = df.rename_axis(self._label, axis=1)
        return df

    def predict(self, X):
        if self.cost_function is not None:
            y_prob = self.predict_proba(X)
            y_hat = self.cost_function.predict(y_prob)
            y_hat = y_hat.to_frame(self._label)
            return y_hat

        if self.selected_features is not None:
            X = X[self.selected_features]

        y_hat = self.model.predict(X)
        y_hat = Series(y_hat, index=X.index).to_frame(self._label)
        return y_hat

    def explain(self):
        report = []
        line = '\n\n{}\n\n'.format('-' * 70)
        report.append(str(self.scores_estimate))
        report.append(str(self.scores))
        return line.join(report)
