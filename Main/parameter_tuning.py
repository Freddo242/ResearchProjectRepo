import numpy as np
import time as time
from tqdm import tqdm

from classifiers import QUBOSoftMarginClassifier, SoftMarginKernelClassifier

import cross_validation

def tune_qsvm_parameters(X_train, t_train, B_values, K_values, R_values, gamma_values, kernel_func, k_folds = 10, num_models = 10):
    """
    Function trains QUBOSoftMarginClassifier on every combination of parameters provided
    Metrics are calucalted using k-fold cross validation and averaged from num_models number of models in each fold.
    Returns accuracy, AUROC, times in a (len(B_values), len(K_values), len(R_values), len(gamma_values)) shape matrix.
    """

    accuracy_results = np.zeros((len(B_values), len(K_values), len(R_values), len(gamma_values)))
    auroc_results = np.zeros((len(B_values), len(K_values), len(R_values), len(gamma_values)))
    times = np.zeros((len(B_values), len(K_values), len(R_values), len(gamma_values)))
    
    n = len(B_values) * len(K_values) * len(R_values) * len(gamma_values)
    with tqdm(desc='Progress', total = n) as pbar:

        for i, B in enumerate(B_values):
            for j, K in enumerate(K_values):
                for k, R in enumerate(R_values):
                    for l, gamma in enumerate(gamma_values):
                    
                        t0 = time.time()
                        qubo_model = QUBOSoftMarginClassifier(B, K, R, kernel_func, gamma)
                        acc, auc = cross_validation.qp_cross_validate(X_train, t_train, k_folds, qubo_model, num_models)
                        t1 = time.time()
                        accuracy_results[i, j, k, l] = np.mean(acc)
                        auroc_results[i, j, k, l] = np.mean(auc)
                        times[i, j, k, l] = t1 - t0

                        pbar.update(1)

    return accuracy_results, auroc_results, times
    
def tune_csvm_parameters(X_train, t_train, C_values, gamma_values, kernel_func, k_folds = 10):
    """
    Function trains SoftMarginKernelClassifier on every combination of parameters provided.
    Metrics are calculated using k-fold cross validation. 
    Returns accuracy, AUROC, times in a (len(C_values), len(gamma_values)) shape matrix.
    """
    accuracy_results = np.zeros((len(C_values), len(gamma_values)))
    auroc_results = np.zeros((len(C_values), len(gamma_values)))
    times = np.zeros((len(C_values), len(gamma_values)))

    with tqdm(desc='Progress', total = len(C_values) * len(gamma_values)) as pbar:

        for i, C in enumerate(C_values):
            for j, gamma in enumerate(gamma_values):

                t0 = time.time()
                clf = SoftMarginKernelClassifier(C, kernel_func, gamma)
                acc, auc = cross_validation.qp_cross_validate(X_train, t_train, k_folds, clf, num_models = 1)
                if acc is not None:
                    accuracy_results[i, j] = np.mean(acc)
                    auroc_results[i, j] = np.mean(auc)
                else:
                    accuracy_results[i, j] = acc
                    auroc_results[i, j] = auc

                t1 = time.time()
                times[i, j] = t1 - t0

                pbar.update(1)

    return accuracy_results, auroc_results, times

def main():
    pass

if __name__ == "__main__":
    main()