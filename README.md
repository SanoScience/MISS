# MISS

### | [Paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611978032.7) |

This is a repository containing the code to generate Multiclass Interpretable Scoring Systems (MISS) as the one below:

![MISS](./images/example.png)

### Installation

To use MISS, clone the repository and install all the required libraries:
```shell
pip install -r requirements.txt
```


Then, install risk-slim with multiclass extensions:
```shell
cd risk-slim
pip install -e .
```

### Usage
You can run the example MISS training with:
```shell
cd miss
python miss_example.py
```

This will create a multiclass scoring system based on the iris dataset.

You can train your own scoring systems with scikit-learn compatible api:
```python
from miss.models import MISSClassifier

mcrsc = MISSClassifier(
    mc_l0_min=0,
    mc_l0_max=3,
    max_coefficient=5,
    max_intercept=10
)

x_train = #... load dataset with binary features
y_train = #... pandas dataframe with 0, ..., K-1 values

mcrsc.fit(x_train, y_train)
```


### References
Please consider citing our paper:
```
@inproceedings{grzeszczyk2024miss,
  title={MISS: Multiclass Interpretable Scoring Systems},
  author={Grzeszczyk, Michal K and Trzci{\'n}ski, Tomasz and Sitek, Arkadiusz},
  booktitle={Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
  pages={55--63},
  year={2024},
  organization={SIAM}
}
```


The implementation of risk-slim is taken from the original [risk-slim repository](https://github.com/ustunb/risk-slim). 
We have broadened the implementation to enable Multiclass (mc) scoring systems generation.

Among the most important papers that helped during the implemenation of this project we have to name:
```
Ustun, Berk, and Cynthia Rudin. "Learning optimized risk scores." Journal of Machine Learning Research 20.150 (2019): 1-75.

Pajor, Arkadiusz, et al. "Effect of feature discretization on classification performance of explainable scoring-based machine learning model." International Conference on Computational Science. Cham: Springer International Publishing, 2022.
```
