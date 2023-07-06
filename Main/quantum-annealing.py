import numpy as np
import pandas as pandas
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

    #Set up sampler
    sampler = EmbeddingComposite(DWaveSampler())
    print("sampler initialised")
    #Set up qubo problem. Going to use from_qubo since from_numpy_array didn't work too well
    N = model.Q.shape[0]
    qubo = np.triu(model.Q + np.identity(N) * model.q)
    qubo_dict = {(i, j): qubo[i, j] for i in range(N) for j in range(N)} 
    print("qubo dict created")

    #bqm = BinaryQuadraticModel.from_qubo(qubo_dict)
    #print("bqm model created")
    #Sample the results
    sample_set = sampler.sample_qubo(qubo_dict, num_reads = 100)
    print("sampler sampled")
    return sample_set


def main():

    #Get data from file
    #create QUBO SVM classifier
    #Train using quantum fitness function which uses bqm
    filepath = 'synth_data/synth_0.3.csv'

    data = np.loadtxt(filepath, delimiter = ',')
    X = data[:, :-1]
    t = data[:, -1]

    X_train, X_test, t_train, t_test = train_test_split(X, t, train_size = 0.1, shuffle = True, stratify = t)
    t_train = t_train.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    B = 2
    K = 3
    R = 1
    kernel_func = rbf_kernel
    gamma = 4

    QSVM = QUBOSoftMarginClassifier(B, K, R, kernel_func, gamma)
    QSVM = QSVM.make_QUBO_problem(X_train, t_train)
    print("problem created. Size: ", QSVM.Q.shape[0])
    samples = quantum_fit(X_train, t_train, QSVM)
    df = samples.to_pandas_dataframe()
    df.to_csv('QA_results/testing_samples.csv')
    
    pass


if __name__ == "__main__":
    main()