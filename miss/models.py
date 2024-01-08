import abc
from typing import Dict, Optional

import numpy as np
import pandas as pd
import prettytable as pt
import warnings
import riskslim
from riskslim.defaults import INTERCEPT_NAME
from riskslim.utils import check_data
from riskslim.coefficient_set import MCCoefficientSet
from riskslim.loss_functions.ce_loss import softmax


class RiskSLIMBase(abc.ABC):
    """
    Abstract base class for classes that wrap original RiskSLIM implementation.
    """

    def __init__(
            self,
            l0_min,
            l0_max,
            max_coefficient,
            max_intercept,
            is_multiclass=False,
            c0_value=1e-6,
            w_pos=1.00,

            # MIP Formulation
            drop_variables=True,
            tight_formulation=True,
            include_auxillary_variable_for_objval=True,
            include_auxillary_variable_for_L0_norm=True,  # noqa

            # LCPA Settings
            max_runtime=300,
            max_tolerance=1e-6,
            display_cplex_progress=False,
            loss_computation="normal",
            chained_updates_flag=True,
            initialization_flag=False,
            initial_bound_updates=True,
            add_cuts_at_heuristic_solutions=True,

            # LCPA Rounding Heuristic
            round_flag=True,
            polish_rounded_solutions=True,
            rounding_tolerance=float("inf"),
            rounding_start_cuts=0,
            rounding_start_gap=float("inf"),
            rounding_stop_cuts=20000,
            rounding_stop_gap=0.2,

            # LCPA Polishing Heuristic
            polish_flag=True,
            polishing_tolerance=0.1,
            polishing_max_runtime=10,
            polishing_max_solutions=5,
            polishing_start_cuts=0,
            polishing_start_gap=float("inf"),
            polishing_stop_cuts=float("inf"),
            polishing_stop_gap=5,

            # Internal parameters
            purge_loss_cuts=False,
            purge_bound_cuts=False,

            # CPLEX settings
            cplex_randomseed=0,
            cplex_mipemphasis=0,
            cplex_mipgap=np.finfo("float").eps,
            cplex_absmipgap=np.finfo("float").eps,
            cplex_integrality_tolerance=np.finfo("float").eps,
            cplex_repairtries=20,
            cplex_poolsize=100,
            cplex_poolrelgap=float("nan"),
            cplex_poolreplace=2,
            cplex_n_cores=1,
            cplex_nodefilesize=120 * 1024,

            # CPA initialization settings
            init_type="cvx",
            init_display_progress=False,
            init_display_cplex_progress=False,
            init_save_progress=False,
            init_update_bounds=True,
            init_max_runtime=300,
            init_max_runtime_per_integration=15,
            init_max_coefficient_gap=0.49,
            init_min_iterations_before_coefficient_gap_check=250,
            init_max_iterations=10000,
            init_max_tolerance=1e-5,

            # Initialization settings
            init_use_rounding=True,
            init_rounding_max_runtime=30,
            init_rounding_max_solutions=5,
            init_use_sequential_rounding=True,
            init_sequential_rounding_max_runtime=30,
            init_sequential_rounding_max_solutions=5,
            init_polishing_after=True,
            init_polishing_max_runtime=30,
            init_polishing_max_solutions=5,

            # Multiclass Settings
            mc_l0_min=0,
            mc_l0_max=4,
            mc_c0_value=1e-6
    ):
        """
        :param l0_min: minimal number of features that should be included in a risk score system.
        :param l0_max: maximal number of features that should be included in a risk score system.
        :param max_coefficient: maximal absolute value that single coefficient should have.
        :param max_intercept: maximal absolute value of an intercept of final model.
        :param is_multiclass: is multiclass model flag.
        """

        self._l0_min = l0_min
        self._l0_max = l0_max
        self._mc_l0_min = mc_l0_min
        self._mc_l0_max = mc_l0_max
        self._max_coefficient = max_coefficient
        self._max_intercept = max_intercept
        self._is_multiclass = is_multiclass

        self._settings = {
            "is_multiclass": is_multiclass,
            "c0_value": c0_value,
            "w_pos": w_pos,
            "drop_variables": drop_variables,
            "tight_formulation": tight_formulation,
            "include_auxillary_variable_for_objval": include_auxillary_variable_for_objval,
            "include_auxillary_variable_for_L0_norm": include_auxillary_variable_for_L0_norm,

            "max_runtime": max_runtime,
            "max_tolerance": max_tolerance,
            "display_cplex_progress": display_cplex_progress,
            "loss_computation": loss_computation,
            "chained_updates_flag": chained_updates_flag,
            "initialization_flag": initialization_flag,
            "initial_bound_updates": initial_bound_updates,
            "add_cuts_at_heuristic_solutions": add_cuts_at_heuristic_solutions,

            "round_flag": round_flag,
            "polish_rounded_solutions": polish_rounded_solutions,
            "rounding_tolerance": rounding_tolerance,
            "rounding_start_cuts": rounding_start_cuts,
            "rounding_start_gap": rounding_start_gap,
            "rounding_stop_cuts": rounding_stop_cuts,
            "rounding_stop_gap": rounding_stop_gap,

            "polish_flag": polish_flag,
            "polishing_tolerance": polishing_tolerance,
            "polishing_max_runtime": polishing_max_runtime,
            "polishing_max_solutions": polishing_max_solutions,
            "polishing_start_cuts": polishing_start_cuts,
            "polishing_start_gap": polishing_start_gap,
            "polishing_stop_cuts": polishing_stop_cuts,
            "polishing_stop_gap": polishing_stop_gap,

            "purge_loss_cuts": purge_loss_cuts,
            "purge_bound_cuts": purge_bound_cuts,

            "cplex_randomseed": cplex_randomseed,
            "cplex_mipemphasis": cplex_mipemphasis,
            "cplex_mipgap": cplex_mipgap,
            "cplex_absmipgap": cplex_absmipgap,
            "cplex_integrality_tolerance": cplex_integrality_tolerance,
            "cplex_repairtries": cplex_repairtries,
            "cplex_poolsize": cplex_poolsize,
            "cplex_poolrelgap": cplex_poolrelgap,
            "cplex_poolreplace": cplex_poolreplace,
            "cplex_n_cores": cplex_n_cores,
            "cplex_nodefilesize": cplex_nodefilesize,

            "init_type": init_type,
            "init_display_progress": init_display_progress,
            "init_display_cplex_progress": init_display_cplex_progress,
            "init_save_progress": init_save_progress,
            "init_update_bounds": init_update_bounds,
            "init_max_runtime": init_max_runtime,
            "init_max_runtime_per_integration": init_max_runtime_per_integration,
            "init_max_coefficient_gap": init_max_coefficient_gap,
            "init_min_iterations_before_coefficient_gap_check": init_min_iterations_before_coefficient_gap_check,
            "init_max_iterations": init_max_iterations,
            "init_max_tolerance": init_max_tolerance,

            "init_use_rounding": init_use_rounding,
            "init_rounding_max_runtime": init_rounding_max_runtime,
            "init_rounding_max_solutions": init_rounding_max_solutions,
            "init_use_sequential_rounding": init_use_sequential_rounding,
            "init_sequential_rounding_max_runtime": init_sequential_rounding_max_runtime,
            "init_sequential_rounding_max_solutions": init_sequential_rounding_max_solutions,
            "init_polishing_after": init_polishing_after,
            "init_polishing_max_runtime": init_polishing_max_runtime,
            "init_polishing_max_solutions": init_polishing_max_solutions,

            "mc_l0_min": mc_l0_min,
            "mc_l0_max": mc_l0_max,
            "mc_c0_value": mc_c0_value
        }

    @abc.abstractmethod
    def optimality_gap(self):
        raise NotImplementedError()

    @staticmethod
    def _validate_fit(x: pd.DataFrame, y: pd.DataFrame):
        if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
            raise TypeError("only pandas DataFrames are allowed")

    def _load_data_from_dataframe(self, df, sample_weights_df=None, fold_idx_df=None, fold_num=0):

        # IMPORTANT: below code is copied from the original riskslim implementation (check load_data_from_csv func).

        raw_data = df.to_numpy()
        data_headers = list(df.columns.values)
        n = raw_data.shape[0]

        # setup Y vector and Y_name
        y_col_idx = [0]
        y = raw_data[:, y_col_idx]
        y_name = data_headers[y_col_idx[0]]

        # DO NOT CHANGE 0 to -1 in MC setting
        if not self._is_multiclass:
            y[y == 0] = -1
        else:
            # One hot encode
            y = y.flatten().astype(int)
            z = np.zeros((y.size, (y.max() + 1)))
            z[np.arange(y.size), y] = 1
            y = z

        # setup X and X_names
        x_col_idx = [j for j in range(raw_data.shape[1]) if j not in y_col_idx]
        x = raw_data[:, x_col_idx]
        variable_names = [data_headers[j] for j in x_col_idx]

        # insert a column of ones to X for the intercept
        x = np.insert(arr=x, obj=0, values=np.ones(n), axis=1)
        variable_names.insert(0, INTERCEPT_NAME)

        if sample_weights_df is None:
            sample_weights = np.ones(n)
        else:
            sample_weights = sample_weights_df.to_numpy()

        data = {
            'X': x,
            'Y': y,
            'variable_names': variable_names,
            'outcome_name': y_name,
            'sample_weights': sample_weights,
        }

        # load folds
        if fold_idx_df is not None:
            fold_idx = fold_idx_df.values.flatten()
            k = max(fold_idx)
            all_fold_nums = np.sort(np.unique(fold_idx))
            assert len(fold_idx) == n, "dimension mismatch: read %r fold indices (expected N = %r)" % (len(fold_idx), n)
            assert np.all(all_fold_nums == np.arange(1, k + 1)), "folds should contain indices between 1 to %r" % k
            assert fold_num in np.arange(0, k + 1), "fold_num should either be 0 or an integer between 1 to %r" % k
            if fold_num >= 1:
                # test_idx = fold_num == fold_idx
                train_idx = fold_num != fold_idx
                data['X'] = data['X'][train_idx,]
                data['Y'] = data['Y'][train_idx]
                data['sample_weights'] = data['sample_weights'][train_idx]

        assert check_data(data, is_multiclass=self._is_multiclass)
        return data


class RiskSLIMClassifier(RiskSLIMBase):
    """
    RiskSLIM original implementation wrapper, capable of risk estimation and binary classification.
    Extend its interface by alignment to scikit-learn API. Work with pandas DataFrames only, as column names are
    used in a resulting risk score system.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Fit model to data.
        :param x: feature vectors to which model should be fit.
        :param y: column of values (0 or 1) of target feature to which model should be fit.
        """
        self._validate_fit(x, y)
        if x.shape[-1] < self._l0_max:
            self._l0_max = x.shape[-1]
        concatenated = np.concatenate([y, x], axis=1)
        attr_names = list(x.columns)
        class_name = f"is_{y.columns[0]}"
        dataframe = pd.DataFrame(
            concatenated,
            columns=[class_name, *attr_names])
        class_data = self._load_data_from_dataframe(dataframe)
        coefficient_set = riskslim.CoefficientSet(
            variable_names=class_data['variable_names'],
            lb=-self._max_coefficient,
            ub=self._max_coefficient,
            sign=0)
        coefficient_set.update_intercept_bounds(
            X=class_data['X'],
            y=class_data['Y'],
            max_offset=self._max_intercept)
        constraints = {
            'L0_min': self._l0_min,
            'L0_max': self._l0_max,
            'coef_set': coefficient_set
        }
        model_info, mip_info, lcpa_info = riskslim.run_lattice_cpa(
            class_data, constraints, self._settings)
        model_info["coef_set"] = None
        self._model = (model_info, class_data)

    def predict_proba(self, x: pd.DataFrame):
        """
        Predict a probability (interpreted as risk) of a positive class.
        :param x: feature vectors for which the probability of target feature should be predicted.
        """
        intercept_value = self._model[0]["solution"][0]
        attr_points = self._model[0]["solution"][1:]
        scores = np.sum(np.multiply(attr_points, x), axis=1)
        probas = 1 / (1 + np.exp(-(intercept_value + scores)))
        return probas.values

    def predict(self, x) -> np.ndarray:
        """
        Predict a class of target feature (0 or 1).
        For probabilities higher or equal to 0.5 a positive class is assumed.
        :param x: feature vectors for which the target feature should be predicted.
        :return: vector of predicted classes.
        """
        return np.where(self.predict_proba(x) >= 0.5, 1, 0)

    def get_post_training_params(self) -> dict:
        """
        Return parameters of a model after a training.
        :return: dictionary containing an intercept value, features, and coefficients of risk score system,
            and loss value of cost function at which the training has finished.
        """
        if self._model is None:
            raise RuntimeError("model must be fit before")

        feature_coefficients = sorted([
            (feature_name, coefficient)
            for feature_name, coefficient in zip(
                self._model[1]["variable_names"][1:], self._model[0]["solution"][1:])
            if coefficient != 0
        ], key=lambda item: -item[1])

        return {
            "intercept": self._model[0]["solution"][0],
            "features_coefficients": feature_coefficients,
            "loss_value": self._model[0]["loss_value"]
        }

    def optimality_gap(self):
        return self._model[0]["optimality_gap"]

    def to_latex(self, show_ommitted_variables=False):
        if self._model is None:
            raise RuntimeError("model must be fit before")
        post_params = self.get_post_training_params()
        intercept = post_params["intercept"]
        coefficients = self._model[0]["solution"][1:]
        variable_names = self._model[1]["variable_names"][1:]

        st = r"\begin{tabular}{cc}\hline" + "\n"
        st += r"\textbf{feature} & \textbf{points}"
        st += "\\\\\\hline\n"

        for i, var_name in enumerate(variable_names):
            if not show_ommitted_variables and coefficients[i] != 0:
                st += r"$\bm{" + var_name + "}$" + f" & {round(coefficients[i])}"
                st += "\\\\\n"
        st += "\\hline\n"

        st += "Score: & = ...."
        st += "\\\\\\hline\n"

        st += r"\multicolumn{2}{c}{\textbf{risk:} $\frac{1}{1 +  exp(" + f"{round(intercept)}" + r" - score)}$}\\\hline"

        st += '\n\\end{tabular}\n'

        st = st.replace("_", "\\_").replace("<=", "\leq")

        return st


class OvRRiskSLIMClassifier(RiskSLIMBase):
    """
    Extension to the original RiskSLIM implementation which allows for a multi-class classification.
    Internally it builds multiple `RiskSLIMClassifier` instances and multiple binary classifications are performed
    (one vs rest), and the class with the highest probability is chosen as a result of a classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submodels: Optional[Dict[str, RiskSLIMClassifier]] = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Fit model to data.
        :param x: feature vectors to which model should be fit.
        :param y: column of values of target feature to which model should be fit.
        """
        self._validate_fit(x, y)
        unique_class_values = np.unique(y)
        class_name = y.columns[0]
        submodels = {}
        for class_value in unique_class_values:
            if class_value == 0 and len(unique_class_values) == 2:
                continue
            specific_class_name = f"{class_name}_{class_value}"
            y_specific_class = pd.DataFrame(
                np.where(y != class_value, 0, 1).reshape(-1, 1),
                columns=[specific_class_name])
            submodel = RiskSLIMClassifier(
                l0_min=self._l0_min,
                l0_max=self._l0_max,
                max_coefficient=self._max_coefficient,
                max_intercept=self._max_intercept,
                **self._settings)
            submodel.fit(x, y_specific_class)
            submodels[class_value] = submodel
        self._submodels = submodels

    def predict(self, x) -> np.ndarray:
        """
        Predict a class of target feature.
        :param x: feature vectors for which the target feature should be predicted.
        :return: vector of predicted classes.
        """
        classes = []
        for class_value in self._submodels.keys():
            classes.append(class_value)

        y_pred_proba = self.predict_proba(x)
        y_pred_proba = y_pred_proba.T

        max_proba_idxs = np.argmax(y_pred_proba, axis=0)

        if len(self._submodels.keys()) == 1:  # binary classification
            return max_proba_idxs
        predicted = np.array([
            classes[max_proba_idx] for max_proba_idx in max_proba_idxs
        ])
        return predicted

    def predict_proba(self, x) -> np.ndarray:
        classes = []
        probas = []
        for class_value, model in self._submodels.items():
            classes.append(class_value)
            probas.append(model.predict_proba(x).reshape(1, -1))
        concatenated = np.concatenate(probas).T
        if len(self._submodels.keys()) == 1:  # binary classification
            concatenated = concatenated.flatten()
            concatenated = np.array([*zip((1 - concatenated), concatenated)])
        else:
            concatenated /= concatenated.sum(axis=1).reshape((concatenated.shape[0], -1))
        return concatenated

    def get_post_training_params(self):
        """
        Return parameters of each model built internally after a training.
        :return: dictionary containing an intercept value, features, and coefficients of risk score system,
            and loss value of cost function at which the training has finished for each model built internally.
        """
        if self._submodels is None:
            raise RuntimeError("model must be fit before")
        return {
            class_value: submodel.get_post_training_params()
            for class_value, submodel in self._submodels.items()
        }

    def optimality_gap(self):
        optimality_gaps = []
        for _, model in self._submodels.items():
            optimality_gaps.append(model.optimality_gap())
        return np.mean(np.array(optimality_gaps))

    def to_latex(self):
        latex_list = []
        classes = []
        for class_value, model in self._submodels.items():
            classes.append(class_value)
            latex_list.append(model.to_latex())

        st = ""
        st += r"\begin{table}" + "\n"

        st += r"\caption{Global caption}"
        st += "\n"

        for latex, c in zip(latex_list, classes):
            latex = latex.replace("risk:", f"risk {c}:")

            st += r"\begin{subtable}[t]{.33\textwidth}" + "\n" r"\caption{OvR}" + "\n" + r"\adjustbox{width =\textwidth}{"

            st += latex
            st += "}\n"
            st += r"\end{subtable}"
            st += "\n"

        st += r"\end{table}"
        return st


class MISSClassifier(RiskSLIMBase):
    """
    MISS classification model.
    Works with pandas DataFrames only, as column names are
    used in a resulting risk score system.

    :param mc_l0_min: minimum number of features in the final MCRiskSLIM model
    :param mc_l0_max: maximum number of features in the final MCRiskSLIM model
    :param mc_c0_value: penalization term for adding feature to the model

    :param l0_min: minimum number of coefficients in the final MCRiskSLIM model. This param doesn't have any effect when
    c0_value is 0.
    :param l0_max: maximum number of coefficients in the final MCRiskSLIM model. This param doesn't have any effect when
    c0_value is 0.
    :param c0_value: penalization term for adding coefficient to the model. The default value is 0 as we want to
    penalize features not coefficients
    """

    def __init__(self, mc_l0_min, mc_l0_max, l0_min=0, l0_max=0, c0_value=0, mc_c0_value=1e-6, oversampler=None,
                 feature_selector=None, **kwargs):
        super().__init__(l0_min=l0_min, l0_max=l0_max, mc_l0_min=mc_l0_min, mc_l0_max=mc_l0_max,
                         mc_c0_value=mc_c0_value, is_multiclass=True,
                         c0_value=c0_value, polishing_stop_gap=0.4, **kwargs)

        if l0_max > 0 and c0_value == 0:
            warnings.warn(f"l0_max>0 ({l0_max}) but c0_value=0 so this is not going to have any effect on the model")

        self._c0_value = c0_value
        self._oversampler = oversampler
        self._feature_selector = feature_selector
        self._model = None
        self._rho = None
        self.num_classes = None
        self.class_names = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, class_names=None):
        """
        Fit model to data.
        :param x: feature vectors to which model should be fit.
        :param y: column of integer values (0,1,2,3...) of target feature to which model should be fit.
        :param class_names: list of strings containing names of target variable.
        """
        self.class_names = class_names
        self._validate_fit(x, y)

        if self._oversampler is not None:
            print("Balancing dataset...")
            x, y = self._oversampler.fit_resample(x, y)

        if self._feature_selector is not None:
            self._feature_selector._miss_params = {"max_coefficient": self._max_coefficient,
                                                   "max_intercept": self._max_intercept}
            self._feature_selector.fit(x, y)
            cols = self._feature_selector.get_support(indices=True)
            x = x.iloc[:, cols]

        unique_class_values = np.unique(y)
        num_classes = len(unique_class_values)
        self.num_classes = num_classes
        class_name = y.columns[0]
        concatenated = np.concatenate([y, x], axis=1)
        attr_names = list(x.columns)
        dataframe = pd.DataFrame(
            concatenated,
            columns=[class_name, *attr_names])
        class_data = self._load_data_from_dataframe(dataframe)
        coefficient_set = MCCoefficientSet(
            variable_names=class_data['variable_names'],
            lb=-self._max_coefficient,
            ub=self._max_coefficient,
            max_intercept=self._max_intercept,
            sign=0,
            num_classes=num_classes)

        if self._l0_max == 0 and self._c0_value == 0:  # L0 norm not counted so setting l0_max to maximum value:
            self._l0_max = self.num_classes * class_data['X'].shape[-1]

        constraints = {
            'L0_min': self._l0_min,
            'L0_max': self._l0_max,
            'mc_L0_min': self._mc_l0_min,
            'mc_L0_max': self._mc_l0_max,
            'coef_set': coefficient_set
        }
        model_info, mip_info, lcpa_info = riskslim.run_lattice_cpa(
            class_data, constraints, self._settings)
        model_info["coef_set"] = None
        self._model = (model_info, class_data)
        self._rho = self._model[0]["solution"]

    def predict_proba(self, x: pd.DataFrame):
        """
        Predict a probability (interpreted as risk) of a positive class.
        :param x: feature vectors for which the probability of target feature should be predicted.
        """
        x = x.copy()
        if self._feature_selector is not None:
            cols = self._feature_selector.get_support(indices=True)
            x = x.iloc[:, cols]
        x.insert(0, "Bias", 1)
        attr_points = self._rho
        rho = attr_points.reshape((x.shape[-1], -1))
        scores = x.dot(rho)
        probas = softmax(scores)
        return probas.values

    def predict(self, x) -> np.ndarray:
        """
        Predict a class of target feature.
        """
        return np.argmax(self.predict_proba(x), axis=1)

    def get_post_training_params(self) -> dict:
        """
        Return parameters of a model after a training.
        """
        if self._model is None:
            raise RuntimeError("model must be fit before")

        data = self._model[1]
        variable_names = data['variable_names']
        attr_points = np.array(self._rho)
        rho = attr_points.reshape(len(variable_names), -1)

        post_params = {
            "bias": rho[0],
            "features_coefficients": rho,
            "loss_value": self._model[0]["loss_value"]
        }

        if self._feature_selector is not None:
            cols = self._feature_selector.get_support(indices=True)
            post_params.update({
                "selected_features": cols
            })

        return post_params

    def print(self, show_ommitted_variables=False):
        data = self._model[1]
        variable_names = data['variable_names']
        attr_points = np.array(self._rho)
        rho = attr_points.reshape(len(variable_names), -1)
        rho_names = [str(vn) for vn in list(variable_names)]

        total_string = "Score:"

        class_names = [f"Class {i}" for i in range(self.num_classes)] if self.class_names is None else self.class_names
        max_name_col_length = max(len(total_string), max([len(s) for s in rho_names])) + 2
        max_value_col_length = max(5, max([len(s) for s in class_names]))

        m = pt.PrettyTable()
        m.field_names = ["Variable", *class_names]
        m.add_row(["Variable", *class_names])

        m.add_row(['=' * max_name_col_length, *["=" * max_value_col_length for _ in range(self.num_classes)]])

        for i, var_name in enumerate(rho_names):
            if "Intercept" in var_name:
                var_name = "Bias"
            if not show_ommitted_variables and not (rho[i] == 0).all():
                m.add_row([var_name, *[str(p) for p in rho[i]]])

        m.add_row(['=' * max_name_col_length, *["=" * max_value_col_length for _ in range(self.num_classes)]])
        m.add_row([total_string, *["= ....." for _ in range(self.num_classes)]])
        m.header = False
        m.align["Variable"] = "l"
        m.align["Points"] = "r"
        m.align["Tally"] = "r"

        return m

    def __repr__(self):
        return str(self.print())

    def to_latex(self, show_ommitted_variables=False):
        data = self._model[1]
        variable_names = data['variable_names']
        attr_points = np.array(self._rho)
        rho = attr_points.reshape(len(variable_names), -1)
        rho_names = [str(vn) for vn in list(variable_names)]

        class_names = [f"class {i}" for i in range(self.num_classes)] if self.class_names is None else self.class_names

        st = r"\begin{tabular}{l*{" + f"{len(class_names)}" + r"}{c}}\hline" + "\n"
        st += r"\backslashbox{feature}{class}"
        for cn in class_names:
            st += r"& \textbf{" + cn + "}"
        st += "\\\\\\hline\n"

        for i, var_name in enumerate(rho_names):
            if not show_ommitted_variables and not (rho[i] == 0).all():
                if "Intercept" in var_name:
                    var_name = "bias"
                st += r"$\bm{" + var_name + "}$"
                for coefficient in rho[i]:
                    st += f"& {str(coefficient)}"
                st += "\\\\\n"
        st += "\\hline\n"

        st += "Score:"
        for _ in class_names:
            st += "& = ...."
        st += "\\\\\\hline\n"

        st += '\\end{tabular}\n'

        st = st.replace("_", "\\_").replace("<=", "\leq")

        return st

    def optimality_gap(self):
        return self._model[0]["optimality_gap"]
