import sys 
sys.append('../')

import numpy as np 
from sklearn import StratifiedKFold

import kernel_SVM_functions as kSVM
import QUBO_SVM_functions as qSVM
from classifiers import QUBOSoftMarginClassifier
from metric_functions import compute_auc_from_scores, compute_accuracy


def fit_classifier():
    pass

def calibration(B_values, K_values, R_values, gamma_values, kernel_func, k_folds = 10, num_reads = 100):
    """Function runs cross validation with each combination of hyper-parameters given in the lists.
    returns auroc, accuracy, and times for each combination in a (len(B_values), len(K_values), len(R_values), len(gamma_values)) array.
    """
    pass

def QA_cross_validate():
    """Performs K-fold cross validation on a QUBO classifier using Quantum Annealing"""

    #Create folds. For each fold.
    #train test split fold
    #create the QUBO
    #fit the QUBO
    #For each sample: evaluate AUROC, accuracy, time

    pass



def main():
    print("This file contains functions necessary to evaluate QUBOSoftMarginClassifier models using Quantum Annealing in the DWAVE Leap Environment")


if __name__ == "__main__":
    main()