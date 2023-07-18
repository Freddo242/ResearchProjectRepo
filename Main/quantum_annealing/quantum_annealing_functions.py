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


def stratified_sample(t, size = 20):
    """picks indexes for values in arr such that same number of values for each class appear in the sample"""

    assert size % 2 == 0, f"size needs to be an even number please. You gave {size}"

    t = t.flatten()
    poss_indexes = np.arange(len(t))
    pos_group = poss_indexes[t == 1]
    neg_group = poss_indexes[t == -1]

    return np.append(np.random.choice(pos_group, size = int(size / 2), replace = True), np.random.choice(neg_group, size = int(size / 2), replace = True))


def QA_calibration(X_train, t_train, B_values, K_values, R_values, gamma_values, kernel_func, k_folds = 10, num_reads = 100):
    """
    X_train: shape (N, d)
    t_train: shape (N, 1)
    values lists: combination of which we test QA svm on 

    For each combination of values, 
        for 10 stratified samples of 20 datapoints from the training data
        cross validation performed to get an accurate auroc and accuracy score
        
        combination AUROC and Accuracy results averaged over the 10 samples.

    """

    global_wait = True

    auroc = np.zeros((len(B_values), len(K_values), len(R_values), len(gamma_values)))
    accuracy = np.zeros((len(B_values), len(K_values), len(R_values), len(gamma_values)))

    for i, B in enumerate(B_values):
        for j, K in enumerate(K_values):
            for k, R in enumerate(R_values):
                for l, gamma in enumerate(gamma_values):
                    #Make new directory to store results for this hyper-parameter combination.
                    os.mkdir(f'../QA_results/{B, K, R, gamma}')

                    #Setting up the classifier for cross validation
                    qsvmq = QSVMq(B, K, R, kernel_func, gamma)
                    
                    set_auc = []
                    set_acc = []
                    
                    for s in range(10):
                        #Making subdirectory for the sample
                        os.mkdir(f'../QA_results/{B, K, R, gamma}/sample-{s + 1}/')
                        
                        #stratified sample based on t of size 20
                        sample_index = stratified_sample(t_train, 20)

                        X_train_sample = X_train[sample_index]
                        t_train_sample = t_train[sample_index]

                        #filepath to save the fold models from the QA
                        filepath = f'../QA_results/{B, K, R, gamma}/sample-{s + 1}/'

                        auroc_results, accuracy_results = QA_cross_validate(X_train_sample, t_train_sample, qsvmq, filepath)

                        set_auc.append(np.mean(auroc_results))
                        set_acc.append(np.mean(accuracy_results))

                    auroc[i, j, k, l] = np.mean(set_auc)
                    accuracy[i, j, k, l] = np.mean(set_acc)

                    wait = global_wait

                    while wait:
                        x = input('waiting...')

                        if x == 'next':
                            wait = False

                        if x == 'goforit':
                            global_wait = False
                            break

    np.save('../QA_results/QA_auroc', auroc)
    np.save('../QA_results/QA_accuracy', accuracy)

    return auroc, accuracy

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
        classifier = classifier.make_QUBO_problem(X_train_split, t_train_split).fit(X_train_split, t_train_split, filepath, fold = i + 1)

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