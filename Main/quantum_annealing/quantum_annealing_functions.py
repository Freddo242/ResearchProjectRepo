import sys 
sys.append('../')

import numpy as np 
from sklearn import StratifiedKFold

import kernel_SVM_functions as kSVM
import QUBO_SVM_functions as qSVM
from classifiers import QUBOSoftMarginClassifier
from metric_functions import compute_auc_from_scores, compute_accuracy

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler


def fit_classifier():
    pass

def calibration(B_values, K_values, R_values, gamma_values, kernel_func, k_folds = 10, num_reads = 100):
    """Function runs cross validation with each combination of hyper-parameters given in the lists.
    returns auroc, accuracy, and times for each combination in a (len(B_values), len(K_values), len(R_values), len(gamma_values)) array.
    """
    pass

def QA_cross_validate(X_train, t_train, classifier, k_folds = 10, num_reads = 100):
    """Performs K-fold cross validation on a QUBO classifier using Quantum Annealing
    returns accuracy, auroc
    """

    #Create folds. For each fold.
    #train test split fold
    #create the QUBO
    #fit the QUBO
    #For each sample: evaluate AUROC, accuracy
    #Average auroc, accuracy for each sample
    #Average accuracy, auroc for each fold
    #return accuracy, auroc

    skf = StratifiedKFold(n_splits = k_folds)

    accuracy_results = []
    auroc_results = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, t_train)):
        #Creating train and validation sets from the StratifiedKFold class
        X_train_split = X_train[train_idx]
        t_train_split = t_train[train_idx].reshape(-1, 1)
        X_test_split = X_train[test_idx]
        t_test_split = t_train[test_idx].reshape(-1, 1)

        fold_accuracy_results = []
        fold_auroc_results = []

        #Make qubo problem with the fold's training set.
        model = classifier.make_QUBO_problem(X_train_split, t_train_split)

        #make single qubo matrix to solve in dict format.
        N = model.Q.shape[0]
        qubo = np.triu(model.Q + np.identity(N) * model.q)
        qubo_dict = {(i, j): qubo[i, j] for i in range(N) for j in range(N)} 

        #Set up sampler
        sampler = EmbeddingComposite(DWaveSampler())

        #Sample using QA
        sample_set = sampler.sample_qubo(qubo_dict, num_reads = num_reads)

        #For each sample (i.e. alphas/fitted SVM model)
        




    return accuracy, auroc



def main():
    print("This file contains functions necessary to evaluate QUBOSoftMarginClassifier models using Quantum Annealing in the DWAVE Leap Environment")


if __name__ == "__main__":
    main()