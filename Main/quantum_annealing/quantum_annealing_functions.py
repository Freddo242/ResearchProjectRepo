import sys 
import os
sys.path.append('../')

import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import kernel_SVM_functions as kSVM
import QUBO_SVM_functions as qSVM
from quantum_classifier import QSVMq
from metric_functions import compute_auc_from_scores, compute_accuracy

#from dwave.system.composites import EmbeddingComposite
#from dwave.system.samplers import DWaveSampler


def fit_classifier():
    pass

def calibration(X_train, t_train, B_values, K_values, R_values, gamma_values, kernel_func, k_folds = 10, num_reads = 100):
    """Function runs cross validation with each combination of hyper-parameters given in the lists.
    returns auroc, accuracy, and times for each combination in a (len(B_values), len(K_values), len(R_values), len(gamma_values)) array.
    """
    for B in B_values:
        for K in K_values:
            for R in R_values:
                for gamma in gamma_values:
                    #Make new directory to store results for this hyper-parameter combination.
                    os.mkdir(f'{B, K, R, gamma}')

                    qsvmq = QSVMq(B, K, R, kernel_func, gamma)
                    
                    param_auc = []
                    param_acc = []
                    
                    for s in range(10):
                        #Sample 20 datapoints from the dataset
                        #they are now the train test data
                        "Create directory within params directory for the sample's results"
                        "Pass in file path containing both parameter directory and sample directory."
                        #run cross validation to get an auc and accuracy score


    pass

def QA_cross_validate(X_train, t_train, classifier, filepath, k_folds = 10, num_reads = 100):
    """
    X_train: training dataset to be 20 datapoints.
    t_train: corresponding targets of the datapoints.
    classifier: QSVMq classifier from quantum_classifier.py

    Performs stratified K-fold cross validation on a QUBO classifier using Quantum Annealing.

    returns list of accuracy and auroc results for each fold.
    """

    assert X_train.shape[0] == 20, f"X_train does not contain 20 datapoints. Instead {X_train.shape[0]}"

    skf = StratifiedKFold(n_splits = k_folds)

    auroc_results = []
    accuracy_results = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, t_train)):
        #Creating train and validation sets from the StratifiedKFold class
        X_train_split = X_train[train_idx]
        t_train_split = t_train[train_idx].reshape(-1, 1)
        X_test_split = X_train[test_idx]
        t_test_split = t_train[test_idx].reshape(-1, 1)

        #fit classifier. This runs the QA and sets a .top_models attribute containing the top models
        classifier = classifier.fit(X_train_split, t_train_split, filepath, fold = i + 1)

        fold_auc = []
        fold_acc = []

        for alphas in classifier.top_models_arr:
            classifier = classifier.set_model(X_train_split, t_train_split, alphas)

            if len(classifier.support_ids) == 0:
                continue

            preds = classifier.predict(X_test_split)
            scores = classifier.decision_function(X_test_split)

            auc = compute_auc_from_scores(scores, t_test_split)
            acc = compute_accuracy(preds, t_test_split)

            fold_auc.append(auc)
            fold_acc.append(acc)

        if len(fold_auc) == 0:
            auroc_results.append()
            continue

        auroc_results.append(np.mean(fold_auc))
        accuracy_results.append(np.mean(fold_acc))

        

    return auroc_results, accuracy_results


def main():
    print("This file contains functions necessary to evaluate QUBOSoftMarginClassifier models using Quantum Annealing in the DWAVE Leap Environment")


if __name__ == "__main__":
    main()