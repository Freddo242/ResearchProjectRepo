import numpy as np
from sklearn.model_selection import train_test_split

from classifiers import QUBOSoftMarginClassifier
from kernel_SVM_functions import rbf_kernel

from dimod import BinaryQuadraticModel
from dwave.system.samplers import DWaveSampler

def quantum_fit(X_train, t_train, model):
    """
    Uses DWave's 
    """

    assert model.ObjectiveFunction, "Objective function not set. Run make_QUBO_problem method"




def main():

    #Get data from file
    #create QUBO SVM classifier
    #Train using quantum fitness function which uses bqm
    filepath = 'synth_data/synth_0.3.csv'

    data = np.loadtxt(filepath, delimiter = ',')
    X = data[:, :-1]
    t = data[:, -1]

    X_train, X_test, t_train, t_test = train_test_split(X, t, train_size = 0.4, shuffle = True, stratify = t)
    t_train = t_train.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    B = 2
    K = 3
    R = 1
    kernel_func = rbf_kernel
    gamma = 4

    QSVM = QUBOSoftMarginClassifier(B, K, R, kernel_func, gamma)
    QSVM = QSVM.make_QUBO_problem(X_train, t_train)

    print(QSVM.Q.shape, QSVM.q.shape)

    pass


if __name__ == "__main__":
    main()