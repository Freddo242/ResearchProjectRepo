import sys
sys.path.append('../')

import numpy as np
np.random.seed(314159)

from sklearn.model_selection import train_test_split
from quantum_annealing_functions import QA_calibration
from kernel_SVM_functions import rbf_kernel


def main():

    train_size = 0.4
    filename = 'synth_0.3.csv'

    #Preparing the training data
    data = np.loadtxt(f'../synth_data/{filename}', delimiter = ',')
    X = data[:, :-1]
    t = data[:, -1]

    X_train, X_test, t_train, t_test = train_test_split(X, t, train_size = train_size, shuffle = True, stratify = t)
    t_train = t_train.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    #Preparing model hyper-parameter lists
    B_values = [2, 3]
    K_values = [3, 4]
    R_values = [1, 3]
    gamma_values = [2, 4]
    kernel_func = rbf_kernel

    auroc, accuracy = QA_calibration(X_train, t_train, B_values, K_values, R_values, gamma_values, kernel_func)


if __name__ == "__main__":
    main()