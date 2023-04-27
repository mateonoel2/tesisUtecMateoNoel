import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from refit_strategy import refit_strategy
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import datetime

# load data
df = pd.read_csv("processed_models\2014-08-01test.csv")

# arrive speeds
x = []
# arrive times
y = []

for data in df["0"]:
    data = eval(data)
    start = [s.replace("MTA_", "") for s in data[0]]
    stop = [s.replace("MTA_", "") for s in data[1]]
    x.extend(list(zip(start, stop, data[2], data[3], data[4])))
    y.extend(data[5])

print(len(x), len(y))

# kernel RBF (radial basis function)
tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
]

scores = ["precision", "recall"]

grid_search = GridSearchCV(
    SVC(), tuned_parameters, scoring=scores, refit=refit_strategy
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# y_pred = grid_search.predict(x_test)
# print(classification_report(y_test, y_pred))