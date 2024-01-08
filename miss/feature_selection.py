from sklearn.base import MetaEstimatorMixin, BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np
from models import MISSClassifier


class RFA(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """
    Recursive Feature Aggregation method building multiple MISS models with 1 feature. The first MISS
    model is trained on all features and selected is added to the pool of supported features and
    removed from the feature set. The next model is trained on reduced feature set and the procedure is repeated until
    the desired number of features has been selected.
    """

    def _get_support_mask(self):
        return self.support_

    def __init__(self, n_features_to_select=None, step=1):
        self.n_features_to_select = n_features_to_select
        self._miss_params = None
        self.models = []
        self._step = step

    @property
    def miss_params(self):
        return self._miss_params

    @miss_params.setter
    def miss_params(self, value):
        self._miss_params = value

    def fit(self, X, y, **fit_params):
        return self._fit(X, y, **fit_params)

    def _fit(self, X, y, **fit_params):
        if self._miss_params is None:
            raise ValueError("NO params for MISS! Initialize params before fitting.")
        n_features = X.shape[1]
        support_ = np.zeros(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)

        if n_features <= self.n_features_to_select:
            support_ = np.ones(n_features, dtype=bool)
            ranking_ = np.arange(1, n_features+1, dtype=int)
        else:
            while np.sum(support_) < self.n_features_to_select:
                features = np.arange(n_features)[np.logical_not(support_)]

                current_model = MISSClassifier(
                    max_runtime=60,
                    mc_l0_min=1, mc_l0_max=self._step,
                    max_coefficient=self._miss_params["max_coefficient"],
                    max_intercept=self._miss_params["max_intercept"])
                self.models.append(current_model)
                current_model.fit(X.iloc[:, features], y, **fit_params)
                current_model_rho = current_model.get_post_training_params()["features_coefficients"]
                print(current_model)

                classes_num = len(np.unique(y))
                current_model_rho = current_model_rho.reshape((-1, classes_num))
                chosen_features = (np.count_nonzero(current_model_rho, axis=1) > 0)[1:]
                chosen_feature = features[chosen_features]

                support_[chosen_feature] = True
                ranking_[support_] += 1

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        return self
