from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from refit_strategy import refit_strategy
from sklearn.metrics import classification_report

# kernel RBF (radial basis function)
tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
]

scores = ["precision", "recall"]

grid_search = GridSearchCV(
    SVC(), tuned_parameters, scoring=scores, refit=refit_strategy
)
# grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
# y_pred = grid_search.predict(x_test)
# print(classification_report(y_test, y_pred))