import sys
sys.path.insert(0, '~/desktop/research project/researchprojectrepo/main')

import numpy as np
np.random.seed(314159)

from sklearn.model_selection import train_test_split
from quantum_classifier import QSVMq
from quantum_annealing_functions import bagged_models
from kernel_SVM_functions import rbf_kernel


def main():


    datafile = 'synth_0.4.csv'
    datafilepath = f'../synth_data/{datafile}'
    train_size = 0.4

    data = np.loadtxt(datafilepath, delimiter = ',')
    X = data[:, :-1]
    t = data[:, -1]

    X_train, X_test, t_train, t_test = train_test_split(X, t, train_size = train_size, shuffle = True, stratify = t)
    t_train = t_train.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    B = 2
    K = 2
    R = 1
    gamma = 4
    kernel_func = rbf_kernel

    QAClassifier = QSVMq(B, K, R, kernel_func, gamma)
    filename = datafile[:-4] + 'QA_models'

    bagged_models(X_train, t_train, QAClassifier, filename, num_models = 50, bag_size = 10, sample_size = 18)



    pass
    

if __name__ == "__main__":
    main()