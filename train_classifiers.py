from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import pickle

import warnings
warnings.filterwarnings("ignore")


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


# Load the labelled feature vectors
feature_frame = pd.read_csv('results/label_feature_vectors.csv')
feature_frame = feature_frame.set_index('class_name')


X = feature_frame.values[:, :-1]  # features
y = feature_frame.values[:, -1]  # target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Grid search for best hyperparameters for Decision Tree

params = {
    'max_depth': [2, 3, 5, 10, 20, None],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy", "log_loss"]
}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5, n_jobs=-1, verbose=2, scoring = "accuracy")
grid_search_dt.fit(X_train, y_train)

decision_tree_classifier = grid_search_dt.best_estimator_

# Grid search for best hyperparameters for Naive Bayes

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
grid_search_NB = GridSearchCV(GaussianNB(), param_grid=params_NB, cv=5, n_jobs=-1, verbose=2, scoring = "accuracy")
grid_search_NB.fit(X_train, y_train)

naive_bayes_classifier = grid_search_NB.best_estimator_

# Grid search for best hyperparameters for SVM
params_svm = {'kernel': ['rbf', 'linear', 'sigmoid'],
              'C': [0.1, 1, 10],
              'gamma': ['scale', 'auto']}
grid_search_svm = GridSearchCV(SVC(), param_grid=params_svm, cv=5, n_jobs=-1, verbose=2, scoring = "accuracy")
grid_search_svm.fit(X_train, y_train)

svm_classifier = grid_search_svm.best_estimator_

# Grid search for best hyperparameters for Neural Network
params_mlp = {'hidden_layer_sizes': [(100,), (50,), (10,), (5,), (100, 50), (50, 10), (10, 5)], 
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['sgd', 'adam'],
              'alpha': [0.0001, 0.001],
              'learning_rate': ['constant', 'invscaling', 'adaptive']}
grid_search_mlp = GridSearchCV(MLPClassifier(), param_grid=params_mlp, cv=5, n_jobs=-1, verbose=2, scoring = "accuracy")
grid_search_mlp.fit(X_train, y_train)

mlp_classifier = grid_search_mlp.best_estimator_

# Grid search for best hyperparameters for Random Forest
param_rf = {'n_estimators': [10, 50, 100, 200],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [2, 5, 10, 20, None],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': [None, 'sqrt', 'log2']}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid=param_rf, cv=5, n_jobs=-1, verbose=2, scoring = "accuracy")
grid_search_rf.fit(X_train, y_train)

random_forest_classifier = grid_search_rf.best_estimator_

# Best hyperparameters
print('Best hyperparameters:')
print('Decision Tree Classifier:')
print(grid_search_dt.best_params_)
print('Naive Bayes Classifier:')
print(grid_search_NB.best_params_)
print('SVM Classifier:')
print(grid_search_svm.best_params_)
print('Neural Network Classifier:')
print(grid_search_mlp.best_params_)
print('Random Forest Classifier:')
print(grid_search_rf.best_params_)

# Accuracy for all classifiers
print('Accuracy:')
print('Decision Tree Classifier:', grid_search_dt.best_estimator_.score(X_test, y_test))
print('Naive Bayes Classifier:', grid_search_NB.best_estimator_.score(X_test, y_test))
print('SVM Classifier:', grid_search_svm.best_estimator_.score(X_test, y_test))
print('Neural Network Classifier:', grid_search_mlp.best_estimator_.score(X_test, y_test))
print('Random Forest Classifier:', grid_search_rf.best_estimator_.score(X_test, y_test))

# Precision, Recall, F1 Score for all classifiers
print('Precision, Recall, F1 Score:')
print('Decision Tree Classifier:')
y_pred_dtc = grid_search_dt.best_estimator_.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_dtc, average='binary'))
print('Naive Bayes Classifier:')
y_pred_NB = grid_search_NB.best_estimator_.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_NB, average='binary'))
print('SVM Classifier:')
y_pred_svm = grid_search_svm.best_estimator_.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_svm, average='binary'))
print('Neural Network Classifier:')
y_pred_mlp = grid_search_mlp.best_estimator_.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_mlp, average='binary'))
print('Random Forest Classifier:')
y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_rf, average='binary'))

# Save the models
save_model(decision_tree_classifier, 'decision_tree_classifier.pkl')
save_model(naive_bayes_classifier, 'naive_bayes_classifier.pkl')
save_model(svm_classifier, 'svm_classifier.pkl')
save_model(mlp_classifier, 'mlp_classifier.pkl')
save_model(random_forest_classifier, 'random_forest_classifier.pkl')

print('Models saved to decision_tree_classifier.pkl, naive_bayes_classifier.pkl, svm_classifier.pkl, mlp_classifier.pkl, random_forest_classifier.pkl.')