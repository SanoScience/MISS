import sys

sys.path.append("./")

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from calibration import get_ece
from model_param_grids import MISSModel

df = pd.read_csv("iris_binary.csv", index_col=0)
iris = load_iris()
y = pd.DataFrame(df["target"])
x = df.drop(columns=["target"])

skf = StratifiedKFold(n_splits=5)

results = {
    "auc": [],
    "acc": [],
    "prec": [],
    "rec": [],
    "ece": [],
    "optimality_gap": []
}

for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    print("")
    x_train = x.iloc[train_index, :]
    x_test = x.iloc[test_index, :]
    y_train = y.iloc[train_index, :]
    y_test = y.iloc[test_index, :]

    model = MISSModel()
    cls = model.cls(**model.clf_params())
    cls.fit(x_train, y_train)

    y_pred = cls.predict(x_test)
    y_pred_proba = cls.predict_proba(x_test)
    auc = roc_auc_score(y_test, y_pred_proba, average="weighted", multi_class="ovr")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    ece = get_ece(y_pred_proba, y_test.values.ravel(), debias=False, num_bins=15, mode='marginal')

    results["auc"].append(auc)
    results["acc"].append(acc)
    results["prec"].append(prec)
    results["rec"].append(rec)
    results["ece"].append(ece)
    results["optimality_gap"].append(cls.optimality_gap())

    print(cls)

for metric, scores in results.items():
    print(f"{metric} = {np.mean(scores).round(3)} +/- {np.std(scores).round(3)}")
