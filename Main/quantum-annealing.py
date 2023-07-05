import numpy as np
from sklearn.model_selection import train_test_split

from classifiers import QUBOSoftMarginClassifier
from kernel_SVM_functions import rbf_kernel, discriminant_bias
from QUBO_SVM_functions import decode

from dimod import BinaryQuadraticModel
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

def quantum_fit(X_train, t_train, model):
    """
    Uses DWave's sampler and binary quadratic model to solve our qubo.
    """

    assert model.ObjectiveFunction, "Objective function not set. Run make_QUBO_problem method"

    #create the BinaryQuadraticModel

    q = model.q 
    linear = {i: q[i][0] for i in range(q.shape[0])}

    Q = model.Q
    quadratic = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1])}

    bqm = BinaryQuadraticModel(linear, quadratic, 0, "BINARY")
    #Set up sampler
    sampler = EmbeddingComposite(DWaveSampler())
    #Sample the results
    sample_set = sampler.sample(bqm)

    return sample_set


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

    samples = quantum_fit(X_train, t_train, QSVM)
    print(samples, '\n')
    print(type(samples), '\n')
    with open('sample_dump', 'w') as f:
        f.write(samples)
    
    pass


if __name__ == "__main__":
    main()