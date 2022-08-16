"""
This script trains 4 different ML models, prints some metrics for them, and creates a png of each model's ROC curve
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

def Average(list):
    return sum(list) / len(list)
    
# Loading the Wisconsin breast cancer dataset
cancer_data = load_breast_cancer()

sns.set_theme(style = 'darkgrid')

# this counter will help when adding titles to the Seaborn plots
special_counter = 1
model_dict = {1: 'Linear SVC', 2: 'Naive Bayes', 3: 'Logistic Regression', 4: 'XGBoost Classifier'}

# this function will take a model as its argument and cross validate it over 10 folds,
#   then it will print the average accuracy, precision, and recall
#   (if verbose is true, metrics will also print for each individual fold)
def Cross_validate_model(model, verbose = False):
    kf = KFold(n_splits = 10)

    fold_number = 0
    
    # Arrays for holding the basic metrics from each fold
    # (used for calculating averages)
    accuracies = []
    precisions = []
    recalls = []

    # these variables are holding values used for calculating the ROC curve from each fold
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # initializing the ROC plots
    fig, axs = plt.subplots()

    # this loop does the K-fold cross validation
    for train_index, test_index in kf.split(cancer_data.data, cancer_data.target):
        
        # assigning training and testing sets
        X_train = cancer_data.data[train_index]
        y_train = cancer_data.target[train_index]

        X_test = cancer_data.data[test_index]
        y_test = cancer_data.target[test_index]

        # Fitting the passed model to the data, then predicting on the test set
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        # creating a ROC curve for the current fold
        viz = metrics.RocCurveDisplay.from_estimator(model, X_test, y_test, name = f"ROC fold #{fold_number}", alpha=0.3, lw=1, ax=axs)

        # adding ROC metrics to the arrays outside the loop
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # assigning basic metrics from the current fold to some vars
        accuracy = metrics.accuracy_score(y_test, y_predict)
        precision = metrics.precision_score(y_test, y_predict)
        recall = metrics.recall_score(y_test, y_predict)

        if verbose:
            # Printing metrics for individual folds:
            print(f'Metrics for the model, fold#{fold_number}:')
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print('---------------')
            print()

        # adding the current fold's metrics to the arrays outside the loop
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        fold_number += 1

    # adding the chance line to the ROC plot
    ax = sns.lineplot(x = [0,1], y = [0,1], linestyle = '--', lw = 2, color="r", label = 'Chance', alpha = 0.8, ax=axs)

    # calculating the mean ROC curve and adding it to the ROC plot
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    ax = sns.lineplot(x = mean_fpr, y = mean_tpr, color="b", label = "Mean ROC", lw = 2, alpha = 0.8, ax=axs)

    # getting the special counter to assign the right Title to the ROC plot
    global special_counter
    # formatting the ROC plot
    ax.set( xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title= f"{model_dict[special_counter]} ROC curve",)
    ax.legend(loc="lower right")

    # saving the ROC plot to the pngs directory
    plt.savefig(f"./pngs/{model_dict[special_counter].replace(' ', '_')}_ROC_curve.png")

    # printing the average metrics over all the folds
    print(f"Average accuracy for the model over {fold_number} folds: ", Average(accuracies))
    print(f"Average precision for the model over {fold_number} folds: ", Average(precisions))
    print(f"Average recall for the model over {fold_number} folds: ", Average(recalls))
    print('=====================================')
    print()

    special_counter += 1

# Training a Linear Support Vector Classifier
SVMclassifier = LinearSVC(dual=False) # dual = False because number_of_samples > number_of_features
print("Training the SVC classifier...")
Cross_validate_model(SVMclassifier)

# Training a Naive Bayes Classifier
NBclassifier = MultinomialNB()
print("Training Naive Bayes classifier...")
Cross_validate_model(NBclassifier)

# Training a logistic Regression Classifier
LRclassifier = LogisticRegression(max_iter = 4000) # max_iter must be increased in order for the algorithm to converge
print("Training Logistic Regression Classifier")
Cross_validate_model(LRclassifier)
    
# Training a XGBoost classifier
XGBoost_Classifier = xgb.XGBClassifier()
print("Training XGBoost classifier...")
Cross_validate_model(XGBoost_Classifier)