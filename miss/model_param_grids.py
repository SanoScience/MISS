import numpy as np
import re
from models import OvRRiskSLIMClassifier, MISSClassifier
from feature_selection import RFA
from rulelist import RuleList
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def clf_params(self, ds_name=""):
        raise NotImplementedError()

    def clf_param_grid(self, ds_name=""):
        raise NotImplementedError()


class LR(Model):
    class CLS(LogisticRegression):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, X, y, sample_weight=None):
            y = y.values.ravel()
            return super().fit(X, y, sample_weight)

    def __init__(self):
        super(LR, self).__init__(cls=LR.CLS, name="LR")

    def clf_params(self, ds_name=""):
        return {
            "penalty": "l1",
            "solver": "saga",
            "max_iter": 1000
        }

    def clf_param_grid(self, ds_name=""):
        return {
            "C": TrialParams("suggest_float", low=1e-4, high=10., log=True)
        }


class XGB(Model):
    class CLS(XGBClassifier):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, X, y, **kwargs):
            X = self._rename_cols(X)
            return super().fit(X, y, **kwargs)

        def predict(self, X, **kwargs):
            X = self._rename_cols(X)
            return super().predict(X, **kwargs)

        def predict_proba(self, X, **kwargs):
            X = self._rename_cols(X)
            return super().predict_proba(X, **kwargs)

        def _rename_cols(self, X):
            regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                         X.columns.values]
            return X

    def __init__(self):
        super(XGB, self).__init__(cls=XGB.CLS, name="XGB")

    def clf_params(self, ds_name=""):
        return {

        }

    def clf_param_grid(self, ds_name=""):
        return {
            "max_depth": TrialParams("suggest_int", low=1, high=7, step=1.),
            "n_estimators": TrialParams("suggest_int", low=5, high=100)
        }


class OvRRiskSLIM(Model):
    def __init__(self):
        super(OvRRiskSLIM, self).__init__(cls=OvRRiskSLIMClassifier, name="OvRRiskSLIM")

    def clf_params(self, ds_name=""):
        return {
            "max_runtime": 60 * 20,
            "l0_min": 0,
            "l0_max": 10,
            "max_coefficient": 5,
            "max_intercept": 20
        }

    def clf_param_grid(self, ds_name=""):
        return {
        }


class MISSModel(Model):
    def __init__(self):
        super(MISSModel, self).__init__(cls=MISSClassifier, name="MISS")

    def clf_params(self, ds_name=""):
        features = 15
        feature_selector = RFA(n_features_to_select=features)
        return {
            "mc_l0_min": 0,
            "max_runtime": 60 * 90,
            "feature_selector": feature_selector,
            "mc_l0_max": 5,
            "max_coefficient": 5,
            "max_intercept": 20,
        }

    def clf_param_grid(self, ds_name=""):
        return {}


class Unit(Model):
    class CLS(LogisticRegression):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, X, y, sample_weight=None):
            y = y.values.ravel()
            super().fit(X, y, sample_weight)
            self.coef_ = np.sign(self.coef_)
            self.intercept_ = 0
            return self

    def __init__(self):
        super(Unit, self).__init__(cls=Unit.CLS, name="Unit")

    def clf_params(self, ds_name=""):
        return {
            "penalty": "l1",
            "solver": "saga",
            "max_iter": 1000
        }

    def clf_param_grid(self, ds_name=""):
        return {
            "C": TrialParams("suggest_float", low=1e-4, high=10., log=True)
        }


class DecisionTree(Model):
    class CLS(DecisionTreeClassifier):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, X, y, sample_weight=None, check_input=None):
            y = y.values.ravel()
            return super().fit(X, y)

    def __init__(self):
        super(DecisionTree, self).__init__(cls=DecisionTree.CLS, name="DT")

    def clf_params(self, ds_name=""):
        return {
        }

    def clf_param_grid(self, ds_name=""):
        return {
            "criterion": TrialParams("suggest_categorical", choices=["gini", "entropy", "log_loss"]),
            "min_samples_split": TrialParams("suggest_int", low=2, high=10),
            "min_samples_leaf": TrialParams("suggest_int", low=2, high=5)
        }


class RandomForest(Model):
    class CLS(RandomForestClassifier):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, X, y, sample_weight=None, check_input=None):
            y = y.values.ravel()
            return super().fit(X, y)

    def __init__(self):
        super(RandomForest, self).__init__(cls=RandomForest.CLS, name="RF")

    def clf_params(self, ds_name=""):
        return {
        }

    def clf_param_grid(self, ds_name=""):
        return {
            "criterion": TrialParams("suggest_categorical", choices=["gini", "entropy", "log_loss"]),
            "min_samples_split": TrialParams("suggest_int", low=2, high=10),
            "min_samples_leaf": TrialParams("suggest_int", low=2, high=5),
            "n_estimators": TrialParams("suggest_int", low=5, high=100)
        }


class MCRuleList(Model):
    def __init__(self):
        super(MCRuleList, self).__init__(cls=RuleList, name="RuleList")

    def clf_params(self, ds_name=""):
        return {
            "task": 'prediction',
            "target_model": 'categorical',
        }

    def clf_param_grid(self, ds_name=""):
        return {
            "max_depth": TrialParams("suggest_int", low=2, high=7),
            "beam_width": TrialParams("suggest_int", low=50, high=150),
        }


class TrialParams:
    def __init__(self, method_name, **kwargs):
        self._method_name = method_name
        self._kwargs = kwargs

    def apply_on_trial(self, trial, param_name):
        method = getattr(trial, self._method_name)
        return method(param_name, **self._kwargs)
