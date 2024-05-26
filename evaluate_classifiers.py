import pickle
import pandas as pd
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import wilcoxon



decision_tree_classifier = pickle.load(open('decision_tree_classifier.pkl', 'rb'))
gaussian_nb = pickle.load(open('naive_bayes_classifier.pkl', 'rb'))
svc = pickle.load(open('svm_classifier.pkl', 'rb'))
mlp = pickle.load(open('mlp_classifier.pkl', 'rb'))
random_forest = pickle.load(open('random_forest_classifier.pkl', 'rb'))

def biased_classifier(y_test):
    return np.ones(len(y_test))


feature_frame = pd.read_csv('results/label_feature_vectors.csv')
feature_frame = feature_frame.set_index('class_name')


X = feature_frame.values[:, :-1]  # features
y = feature_frame.values[:, -1]  # target

scoring = ['precision', 'recall', 'f1']

dtc_score_list = []
nb_score_list = []
svc_score_list = []
mlp_score_list = []
rf_score_list = []
for _ in range(20):
    score_dtc = cross_validate(decision_tree_classifier, X, y, cv=5, scoring=scoring)
    dtc_score_list.append(score_dtc)
    score_gnb = cross_validate(gaussian_nb, X, y, cv=5, scoring=scoring)
    nb_score_list.append(score_gnb)
    score_svc = cross_validate(svc, X, y, cv=5, scoring=scoring)
    svc_score_list.append(score_svc)
    score_mlp = cross_validate(mlp, X, y, cv=5, scoring=scoring)
    mlp_score_list.append(score_mlp)
    score_rf = cross_validate(random_forest, X, y, cv=5, scoring=scoring)
    rf_score_list.append(score_rf)


dtc_precision = np.concatenate([score['test_precision'] for score in dtc_score_list])
dtc_recall = np.concatenate([score['test_recall'] for score in dtc_score_list])
dtc_f1 = np.concatenate([score['test_f1'] for score in dtc_score_list])

nb_precision = np.concatenate([score['test_precision'] for score in nb_score_list])
nb_recall = np.concatenate([score['test_recall'] for score in nb_score_list])
nb_f1 = np.concatenate([score['test_f1'] for score in nb_score_list])

svc_precision = np.concatenate([score['test_precision'] for score in svc_score_list])
svc_recall = np.concatenate([score['test_recall'] for score in svc_score_list])
svc_f1 = np.concatenate([score['test_f1'] for score in svc_score_list])

mlp_precision = np.concatenate([score['test_precision'] for score in mlp_score_list])
mlp_recall = np.concatenate([score['test_recall'] for score in mlp_score_list])
mlp_f1 = np.concatenate([score['test_f1'] for score in mlp_score_list])

rf_precision = np.concatenate([score['test_precision'] for score in rf_score_list])
rf_recall = np.concatenate([score['test_recall'] for score in rf_score_list])
rf_f1 = np.concatenate([score['test_f1'] for score in rf_score_list])


biased_classifier_precision = np.array([precision_recall_fscore_support(y, biased_classifier(y), average='binary')[0] for _ in range(100)])
biased_classifier_recall = np.array([precision_recall_fscore_support(y, biased_classifier(y), average='binary')[1] for _ in range(100)])
biased_classifier_f1 = np.array([precision_recall_fscore_support(y, biased_classifier(y), average='binary')[2] for _ in range(100)])


# statistics/box plots of evaluation metrics

print('Decision Tree')
print('Precision:', np.mean(dtc_precision), np.std(dtc_precision))
print('Recall:', np.mean(dtc_recall), np.std(dtc_recall))
print('F1:', np.mean(dtc_f1), np.std(dtc_f1))

print('Naive Bayes')
print('Precision:', np.mean(nb_precision), np.std(nb_precision))
print('Recall:', np.mean(nb_recall), np.std(nb_recall))
print('F1:', np.mean(nb_f1), np.std(nb_f1))

print('SVM')
print('Precision:', np.mean(svc_precision), np.std(svc_precision))
print('Recall:', np.mean(svc_recall), np.std(svc_recall))
print('F1:', np.mean(svc_f1), np.std(svc_f1))

print('Neural Network')
print('Precision:', np.mean(mlp_precision), np.std(mlp_precision))
print('Recall:', np.mean(mlp_recall), np.std(mlp_recall))
print('F1:', np.mean(mlp_f1), np.std(mlp_f1))

print('Random Forest')
print('Precision:', np.mean(rf_precision), np.std(rf_precision))
print('Recall:', np.mean(rf_recall), np.std(rf_recall))
print('F1:', np.mean(rf_f1), np.std(rf_f1))

print('Biased Classifier')
print('Precision:', np.mean(biased_classifier_precision), np.std(biased_classifier_precision))
print('Recall:', np.mean(biased_classifier_recall), np.std(biased_classifier_recall))
print('F1:', np.mean(biased_classifier_f1), np.std(biased_classifier_f1))

# Box plots
import matplotlib.pyplot as plt

plt.boxplot([dtc_precision, nb_precision, svc_precision, mlp_precision, rf_precision, biased_classifier_precision])
plt.xticks([1, 2, 3, 4, 5, 6], ['Decision Tree', 'Naive Bayes', 'SVM', 'Neural Network', 'Random Forest', 'Biased Classifier'], rotation=45, ha='right')
plt.ylabel('Precision')
plt.savefig('results/precision.png', bbox_inches='tight', dpi=300)
plt.show()

plt.boxplot([dtc_recall, nb_recall, svc_recall, mlp_recall, rf_recall, biased_classifier_recall])
plt.xticks([1, 2, 3, 4, 5, 6], ['Decision Tree', 'Naive Bayes', 'SVM', 'Neural Network', 'Random Forest', 'Biased Classifier'], rotation=45, ha='right')
plt.ylabel('Recall')
plt.savefig('results/recall.png', bbox_inches='tight', dpi=300)
plt.show()

plt.boxplot([dtc_f1, nb_f1, svc_f1, mlp_f1, rf_f1, biased_classifier_f1])
plt.xticks([1, 2, 3, 4, 5, 6], ['Decision Tree', 'Naive Bayes', 'SVM', 'Neural Network', 'Random Forest', 'Biased Classifier'], rotation=45, ha='right')
plt.ylabel('F1')
plt.savefig('results/f1.png', bbox_inches='tight', dpi=300)
plt.show()

# Wilcoxon signed-rank test
# p-values of pairwise comparisons of classifiers

print('Wilcoxon signed-rank test')

print('Decision Tree vs Naive Bayes')
print('Precision:', wilcoxon(dtc_precision, nb_precision))
print('Recall:', wilcoxon(dtc_recall, nb_recall))
print('F1:', wilcoxon(dtc_f1, nb_f1))

print('Decision Tree vs SVM')
print('Precision:', wilcoxon(dtc_precision, svc_precision))
print('Recall:', wilcoxon(dtc_recall, svc_recall))
print('F1:', wilcoxon(dtc_f1, svc_f1))

print('Decision Tree vs Neural Network')
print('Precision:', wilcoxon(dtc_precision, mlp_precision))
print('Recall:', wilcoxon(dtc_recall, mlp_recall))
print('F1:', wilcoxon(dtc_f1, mlp_f1))

print('Decision Tree vs Random Forest')
print('Precision:', wilcoxon(dtc_precision, rf_precision))
print('Recall:', wilcoxon(dtc_recall, rf_recall))
print('F1:', wilcoxon(dtc_f1, rf_f1))

print('Decision Tree vs Biased Classifier')
print('Precision:', wilcoxon(dtc_precision, biased_classifier_precision))
print('Recall:', wilcoxon(dtc_recall, biased_classifier_recall))
print('F1:', wilcoxon(dtc_f1, biased_classifier_f1))

print('Naive Bayes vs SVM')
print('Precision:', wilcoxon(nb_precision, svc_precision))
print('Recall:', wilcoxon(nb_recall, svc_recall))
print('F1:', wilcoxon(nb_f1, svc_f1))

print('Naive Bayes vs Neural Network')
print('Precision:', wilcoxon(nb_precision, mlp_precision))
print('Recall:', wilcoxon(nb_recall, mlp_recall))
print('F1:', wilcoxon(nb_f1, mlp_f1))

print('Naive Bayes vs Random Forest')
print('Precision:', wilcoxon(nb_precision, rf_precision))
print('Recall:', wilcoxon(nb_recall, rf_recall))
print('F1:', wilcoxon(nb_f1, rf_f1))

print('Naive Bayes vs Biased Classifier')
print('Precision:', wilcoxon(nb_precision, biased_classifier_precision))
print('Recall:', wilcoxon(nb_recall, biased_classifier_recall))
print('F1:', wilcoxon(nb_f1, biased_classifier_f1))

print('SVM vs Neural Network')
print('Precision:', wilcoxon(svc_precision, mlp_precision))
print('Recall:', wilcoxon(svc_recall, mlp_recall))
print('F1:', wilcoxon(svc_f1, mlp_f1))

print('SVM vs Random Forest')
print('Precision:', wilcoxon(svc_precision, rf_precision))
print('Recall:', wilcoxon(svc_recall, rf_recall))
print('F1:', wilcoxon(svc_f1, rf_f1))

print('SVM vs Biased Classifier')
print('Precision:', wilcoxon(svc_precision, biased_classifier_precision))
print('Recall:', wilcoxon(svc_recall, biased_classifier_recall))
print('F1:', wilcoxon(svc_f1, biased_classifier_f1))

print('Neural Network vs Random Forest')
print('Precision:', wilcoxon(mlp_precision, rf_precision))
print('Recall:', wilcoxon(mlp_recall, rf_recall))
print('F1:', wilcoxon(mlp_f1, rf_f1))

print('Neural Network vs Biased Classifier')
print('Precision:', wilcoxon(mlp_precision, biased_classifier_precision))
print('Recall:', wilcoxon(mlp_recall, biased_classifier_recall))
print('F1:', wilcoxon(mlp_f1, biased_classifier_f1))

print('Random Forest vs Biased Classifier')
print('Precision:', wilcoxon(rf_precision, biased_classifier_precision))
print('Recall:', wilcoxon(rf_recall, biased_classifier_recall))
print('F1:', wilcoxon(rf_f1, biased_classifier_f1))



