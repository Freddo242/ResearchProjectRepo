import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import kernel_SVM_functions as kSVM
import QUBO_SVM_functions as qSVM

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler


class QSVMq(object):
    """QUBO soft margin support vector machine, trained using Quantum Annealing """

    def __init__(self, B, K, R, kernel_func, param):
        """
        B: encoding base
        K: dimension of encoded vector
        R: regularisation for alpha.T @ t constraint
        kernel_func: kernel function. e.g. rbf kernel
        param: parameter for kernel function
        """
        self.B = B
        self.K = K
        self.R = R
        self.C = np.sum([B ** i for i in range(K)])
        self.kernel_func = kernel_func
        self.param = param

        self.qubo_dict = None

        self.alphas = None

        self.support_ids = None
        self.support_vectors = None
        self.support_targets = None
        self.support_alphas = None

        self.bias = None

    def make_QUBO_problem(self, X_train, t_train):
        """Given X_train, t_train data to train the model. 
            Creates the qubo dict in order to fit the model using DWave sampler.
        """

        self.Q, self.q = qSVM.make_QUBO_matrices(X_train, t_train, self.kernel_func, self.param, self.B, self.K, self.R)

        N = self.Q.shape[0]
        #Taking the upper triangular of the matrix
        qubo = np.triu(self.Q + np.identity(N) * self.q)
        #Turning the qubo matrix into a dictionary for the sample_qubo() function 
        self.qubo_dict = {(i, j): qubo[i, j] for i in range(N) for j in range(N)} 

        return self
    
    def set_bias(self, bias):
        """Sets the model's bias paramter --- used for the decision function --- to the given input"""
        self.bias = bias
        return self
    
    def fit(self, X_train, t_train, filepath, fold = None, num_reads = 100, num_top_models = 20):
        """
        X_train, t_train: The same data which was used to create the QUBO matrices
        filepath is the directory to save the results in. 
        Fold is for naming the file. If no fold is given, file is saved under its hyper-parameters (B, K, R, gamma)
        Solves the self.qubo_dict using the DWave Sampler and takes 100 reads
        """

        ##setting up sampler
        #sampler = EmbeddingComposite(DWaveSampler())

        ##running QA and taking reads from the lowest energy level.
        #sample_set = sampler.sample_qubo(self.qubo_dict, num_reads = num_reads)
        #sample_df = sample_set.to_pandas_dataframe()

        #if fold:
        #    #reading from csv once we've saved it to avoid a bug with the sample set
        #    sample_df.to_csv(f'{filepath}/{self.B, self.K, self.R, self.param}-f{fold}')
        #    sample_df = pd.read_csv(f'{filepath}/{self.B, self.K, self.R, self.param}-f{fold}')
            
        #else:
        #    sample_df.to_csv(f'{filepath}/{self.B, self.K, self.R, self.param}')
        #    sample_df.read_csv(f'{filepath}/{self.B, self.K, self.R, self.param}')

        #DELETE THIS LINE BEFORE STARTING
        sample_df = pd.read_csv('../QA_results/(2, 3, 1, 2)/sample-1/(2, 3, 1, 2)-f1')

        #sorting by energy to find the top num_models
        sample_df = sample_df.sort_values('energy', ascending = True)
        top_models = sample_df[: num_top_models]
        #turning the encoded alphas into a np array
        encoded_alphas = np.array([list(row[1: -3]) for index, row in top_models.iterrows()])
        
        #Decoding alphas
        self.top_models_arr = np.apply_along_axis(lambda encoded_arr: qSVM.decode(encoded_arr, self.B, self.K), axis = 1, arr = encoded_alphas)

        #Code below is for testing purposes.
        #self.top_models_arr = np.random.randint(0, 2, size = (10, self.K * X_train.shape[0]))
        #test_df = pd.DataFrame(self.top_models_arr)
#
        #if fold:
        #    test_df.to_csv(f'{filepath}/{self.B, self.K, self.R, self.param}-f{fold}')
        #else:
        #    test_df.to_csv(f'{filepath}/{self.B, self.K, self.R, self.param}')

        return self
        
    def set_model(self, X_train, t_train, alphas):
        """Method to set the model with given alphas"""

        #Finding support vectors 
        self.support_ids, self.support_vectors, self.support_targets, self.support_alphas = kSVM.get_support_vectors(X_train, t_train, alphas, self.C) 
        self.bias = kSVM.discriminant_bias(X_train, t_train, alphas, self.C, self.kernel_func, self.param)

        return self
    
    def decision_function(self, X_test):
        return kSVM.score_kSVM(X_test, self.support_vectors, self.support_targets, self.support_alphas, self.kernel_func, self.param, self.bias)

    def predict_proba(self, X_test):
        #Returns the probability of a vector in X_test belonging to the positive class
        scores = self.decision_function(X_test)
        probs = 1 / (1 + np.exp(-scores))
        return probs
    
    def predict(self, X_test):
        """Returns the predicted class for the test data with threshold 0"""
        return np.sign(self.decision_function(X_test))