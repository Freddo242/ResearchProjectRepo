import sys 
import os
sys.path.append('../')

import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from time import time

from quantum_classifier import QSVMq
from metric_functions import compute_auc_from_scores, compute_accuracy


def include_indices(param_combinations, to_skip):
    #Takes in a list of combinations to skip and returns a boolean array of those combinations to include
    indices = []
    for arr in to_skip:
        indices.append(np.array([(param_combinations[i, :, 1] == arr).all() for i in range(param_combinations.shape[0])], dtype=int))

    return ~np.sum(indices, axis = 0).astype(bool)

def stratified_sample(t, size = 20):
    """picks indexes for values in arr such that same number of values for each class appear in the sample"""

    assert size % 2 == 0, f"size needs to be an even number please. You gave {size}"

    t = t.flatten()
    #Getting the indices associated with each class
    poss_indexes = np.arange(len(t))
    pos_group = poss_indexes[t == 1]
    neg_group = poss_indexes[t == -1]

    #returning randomly picked incides from each class with replacement 
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

    auroc = np.load('../QA_results/QA_auroc.npy')
    accuracy = np.load('../QA_results/QA_accuracy.npy')

    param_combinations = np.array([[[i, B], [j, K], [k, R], [l, gamma]] for i, B in enumerate(B_values) 
                                                            for j, K in enumerate(K_values) 
                                                            for k, R in enumerate(R_values)
                                                            for l, gamma in enumerate(gamma_values)])

    #This is an array of the parameter combinations we have already completed. The code is too long to do all at once.
    params_to_skip = np.array([[2, 3, 1, 2], [2, 3, 1, 4], [2, 3, 3, 2], [2, 3, 3, 4], [2, 4, 1, 2], 
    [2, 4, 1, 4], [2, 4, 3, 2], [2, 4, 3, 4], [3, 3, 1, 2], [3, 3, 1, 4], [3, 3, 3, 2], [3, 3, 3, 4]])
    #index of the combinations to include
    include_index = include_indices(param_combinations, params_to_skip)

    param_combinations = param_combinations[include_index, :, :]

    print(param_combinations[:, :, 1])

    for (i, B), (j, K), (k, R), (l, gamma) in param_combinations:

        print(B, K, R, gamma, '\n')

        t0 = time()

        #Make new directory to store results for this hyper-parameter combination.
        os.mkdir(f'../QA_results/{B, K, R, gamma}')

        #Setting up the classifier for cross validation
        qsvmq = QSVMq(B, K, R, kernel_func, gamma)
                    
        set_auc = []
        set_acc = []
                    
        for s in range(5):

            t1 = time()
            #Making subdirectory for the sample
            os.mkdir(f'../QA_results/{B, K, R, gamma}/sample-{s + 1}/')
                        
            #stratified sample based on t of size 20
            sample_index = stratified_sample(t_train, 20)

            X_train_sample = X_train[sample_index]
            t_train_sample = t_train[sample_index]

            #filepath to save the fold models from the QA
            filepath = f'../QA_results/{B, K, R, gamma}/sample-{s + 1}/'

            #Cross validating training and test data for this sample giving us an accurate auroc and accuracy
            auroc_results, accuracy_results = QA_cross_validate(X_train_sample, t_train_sample, qsvmq, filepath)

            #Taking the mean of the metrics across each of the folds
            set_auc.append(np.mean(auroc_results))
            set_acc.append(np.mean(accuracy_results))

            t2 = time()
            print('Sample time', t2 - t1)

        #final metrics are the average over each sample
        auroc[i, j, k, l] = np.mean(set_auc)
        accuracy[i, j, k, l] = np.mean(set_acc)

        np.save('../QA_results/QA_auroc', auroc)
        np.save('../QA_results/QA_accuracy', accuracy)

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

    #Setgs up sklearn StratifiedKFold 
    skf = StratifiedKFold(n_splits = k_folds)

    auroc_results = []
    accuracy_results = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, t_train)):
        #Creating train and validation sets from the StratifiedKFold class
        X_train_split = X_train[train_idx]
        t_train_split = t_train[train_idx].reshape(-1, 1)
        X_test_split = X_train[test_idx]
        t_test_split = t_train[test_idx].reshape(-1, 1)

        no_support_vectors = True

        while no_support_vectors:

            #fit classifier. This runs the QA and sets a .top_models attribute containing the top models
            classifier = classifier.make_QUBO_problem(X_train_split, t_train_split).fit(X_train_split, t_train_split, filepath, fold = i + 1)

            fold_auc = []
            fold_acc = []

            for alphas in classifier.top_models_arr:
                #Iterating through each of the best models and evaluating their performance

                classifier = classifier.set_model(X_train_split, t_train_split, alphas.reshape(-1, 1))

                if len(classifier.support_ids) == 0:
                    #If not support vectors found, discard the model and continue.
                    print("No support vectors found")
                    continue

                no_support_vectors = False

                preds = classifier.predict(X_test_split)
                scores = classifier.decision_function(X_test_split)

                auc = compute_auc_from_scores(scores, t_test_split)
                acc = compute_accuracy(preds, t_test_split)

                fold_auc.append(auc)
                fold_acc.append(acc)

        print("Fold auc, ", np.mean(fold_auc))
        print("Fold acc, ", np.mean(fold_acc))

        auroc_results.append(np.mean(fold_auc))
        accuracy_results.append(np.mean(fold_acc))

    return auroc_results, accuracy_results


def main():
    print("This file contains functions necessary to evaluate QUBOSoftMarginClassifier models using Quantum Annealing in the DWAVE Leap Environment")


if __name__ == "__main__":
    main()