# Author:   Rohith Iyengar
# Date:     04/14/2024
#
# Instructions for running:
# Python needs to be installed.
# Intall required libraries using this:
# pip install numpy pandas scikit-learn xgboost
#
# Run the script using a Python interpreter. This script is tested with Python 3.12

from sklearn import datasets
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

iris = datasets.load_iris()
df = pd.DataFrame(data    = np.c_[iris['data'], iris['target']],
                  columns = iris['feature_names'] + ['target'])

X = df[iris['feature_names']]
y = df['target']

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

def naive_bayes():
    # Naïve Bayes
    from sklearn.naive_bayes import GaussianNB

    reports = []
    auc_scores = []

    # cross-validation loop
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # initialize and train the GaussianNB model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # predictions
        expected = y_test
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # generate and store classification report and AUC scores
        report = metrics.classification_report(expected, predicted, output_dict=True)
        auc = roc_auc_score(expected, probabilities, multi_class='ovr')
        auc_scores.append(auc)
        reports.append(report)

    # function to print average scores
    printReport(reports, auc_scores)

def svc():
    # Support Vector Machines
    from sklearn import svm

    reports = []
    auc_scores = []

    # cross-validation loop
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # initialize and train the SVC model
        model = svm.SVC(kernel='linear', C=0.3, gamma='scale', probability=True)
        model.fit(X_train, y_train)

        # predictions
        expected = y_test
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # generate and store classification report and AUC scores
        report = metrics.classification_report(expected, predicted, output_dict=True)
        auc = roc_auc_score(expected, probabilities, multi_class='ovr')
        auc_scores.append(auc)
        reports.append(report)

    # function to print average scores
    printReport(reports, auc_scores)

def random_forest():
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    reports = []
    auc_scores = []

    # cross-validation loop
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # initialize and train the RandomForestClassifier model
        model = RandomForestClassifier(criterion='gini', n_estimators=50, max_depth=10, max_features='log2', min_samples_leaf=1, min_samples_split=4)
        model.fit(X_train, y_train)

        # predictions
        expected = y_test
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # generate and store classification report and AUC scores
        report = metrics.classification_report(expected, predicted, output_dict=True)
        auc = roc_auc_score(expected, probabilities, multi_class='ovr')
        auc_scores.append(auc)
        reports.append(report)

    # function to print average scores
    printReport(reports, auc_scores)

def k_nearest_neighbors():
    # K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier

    reports = []
    auc_scores = []

    # cross-validation loop
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # initialize and train the KNeighborsClassifier model
        model = KNeighborsClassifier(n_neighbors=4)
        model.fit(X_train, y_train)

        # predictions
        expected = y_test
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # generate and store classification report and AUC scores
        report = metrics.classification_report(expected, predicted, output_dict=True)
        auc = roc_auc_score(expected, probabilities, multi_class='ovr')
        auc_scores.append(auc)
        reports.append(report)

    # function to print average scores
    printReport(reports, auc_scores)

def xgboost():
    # xGBoost
    import xgboost as xgb

    reports = []
    auc_scores = []

    # cross-validation loop
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # initialize and train the XGBoost model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)

        # predictions
        expected = y_test
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # generate and store classification report and AUC scores
        report = metrics.classification_report(expected, predicted, output_dict=True)
        auc = roc_auc_score(expected, probabilities, multi_class='ovr')
        auc_scores.append(auc)
        reports.append(report)

    # function to print average scores
    printReport(reports, auc_scores)

def printReport(reports, auc_scores):
    avg_f1_score = np.mean([report['weighted avg']['f1-score'] for report in reports])
    avg_accurary = np.mean([report['accuracy'] for report in reports])
    avg_auc = np.mean(auc_scores)
    print(f"Average F1-Score: {avg_f1_score}")
    print(f"Average Accuracy: {avg_accurary}")
    print(f"Average ROC AUC: {avg_auc}")

def main():
    
    print("Running Naive Bayes Classifier...")
    naive_bayes()

    print("\nRunning SVM Classifier...")
    svc()

    print("\nRunning Random Forest Classifier...")
    random_forest()
    
    print("\nRunning KNN Classifier...")
    k_nearest_neighbors()
    
    print("\nRunning xGBoost Classifier...")
    xgboost()

if __name__ == "__main__":
    main()