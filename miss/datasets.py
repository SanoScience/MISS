import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


class Dataset:
    def __init__(self):
        self.name = "dataset"
        self.x = None
        self.y = None
        self.features_to_discretize = None
        self.class_names = None

    def get_dataset(self):
        return self.x, self.y

    def describe(self):
        if self.x is None or self.y is None:
            raise ValueError("Dataset was not initialized")

        d = self.x.shape[-1]
        num_clases = self.y.nunique()[0]
        n = self.y.shape[0]
        task_type = "Binary" if num_clases == 2 else "Multiclass"
        classes = self.y.value_counts(sort=False)
        if num_clases == 2:
            nc0 = classes[0]
            nc1 = classes[1]
            nc2 = "-"
            nc3 = ""
        elif num_clases == 3:
            nc0 = classes[0]
            nc1 = classes[1]
            nc2 = classes[2]
            nc3 = ""
        else:
            nc0 = classes[0]
            nc1 = classes[1]
            nc2 = classes[2]
            nc3 = "&" + str(classes[3])
        description = f"{self.name} & {task_type} & {d} & {n} & {nc0} & {nc1} & {nc2}{nc3} & task description & \\cite" + r"{}\\"
        return description


class IrisDataset(Dataset):
    def __init__(self):
        super(IrisDataset, self).__init__()
        self._initialize()

    def _initialize(self):
        self.name = "iris"
        iris = load_iris()
        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.x = pd.DataFrame(iris.data, columns=columns)
        self.y = pd.DataFrame(iris.target, columns=['target'])
        self.features_to_discretize = columns
        self.class_names = iris.target_names


class WineDataset(Dataset):
    def __init__(self):
        super(WineDataset, self).__init__()
        self._initialize()

    def _initialize(self):
        self.name = "wine"
        wine = load_wine()
        self.x = pd.DataFrame(wine.data, columns=wine.feature_names)
        self.y = pd.DataFrame(wine.target, columns=['target'])
        self.features_to_discretize = wine.feature_names
        self.class_names = wine.target_names


class BreastCancerDataset(Dataset):
    def __init__(self):
        super(BreastCancerDataset, self).__init__()
        self._initialize()

    def _initialize(self):
        self.name = "breast"
        bc = load_breast_cancer()
        self.x = pd.DataFrame(bc.data, columns=bc.feature_names)
        self.y = pd.DataFrame(bc.target, columns=['target'])
        self.features_to_discretize = bc.feature_names
        self.class_names = bc.target_names
