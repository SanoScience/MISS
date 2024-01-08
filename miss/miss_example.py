import sys
import pandas as pd

sys.path.append("./")

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from models import MISSClassifier

df= pd.read_csv("iris_binary.csv", index_col=0)
iris = load_iris()
y = pd.DataFrame(df["target"])
x = df.drop(columns=["target"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

mcrsc = MISSClassifier(
    mc_l0_min=0,
    mc_l0_max=3,
    max_coefficient=5,
    max_intercept=10
)

mcrsc.fit(x_train, y_train, class_names=iris.target_names)

y_pred = mcrsc.predict(x_test)
print("IRIS DATASET")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

print(mcrsc)
