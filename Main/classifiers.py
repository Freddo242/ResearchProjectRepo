import numpy as np

import kernel_SVM_functions as kSVM
import QUBO_SVM_functions as qSVM
import simulated_annealing as annealing

class SoftMarginKernelClassifier(object):

    def __init__(self, C, kernel_func, param):
        self.kernel_function = kernel_func
        self.C = C
        self.param = param
        self.support_vectors = None
        self.support_targets = None
        self.support_alphas = None
        self.support_ids = None

    def fit(self, X_train, t_train):
        #Creates self.model, support vectors and bias for later use.
        P, q, G, h, A, b = kSVM.kSVM_matrices(X_train, t_train, self.C, self.kernel_function, self.param)
        self.alphas = kSVM.train_qp(P, q, G, h, A, b)
        if self.alphas is not None:
            self.model = kSVM.get_support_vectors(X_train, t_train, self.alphas, self.C)
            self.support_ids, self.support_vectors, self.support_targets, self.support_alphas = kSVM.get_support_vectors(X_train, t_train, self.alphas, self.C)
            self.bias = kSVM.discriminant_bias(X_train, t_train, self.alphas, self.C, self.kernel_function, self.param)
        return self
    
    def predict(self, X_test):
        return np.sign(self.decision_function(X_test))

    def decision_function(self, X_test):
        return kSVM.score_kSVM(X_test, self.support_vectors, self.support_targets, self.support_alphas, self.kernel_function, self.param, self.bias)

    def predict_proba(self, X_test):
        scores = self.decision_function(X_test)
        probs = 1 / (1 + np.exp(-scores))
        return probs
    

class QUBOSoftMarginClassifier(object):

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

        self.cooling_param = 5
        self.T0 = 10
        self.annealing_iterations = 100
        self.m_rate = 1 / K

        self.ObjectiveFunction = None

        self.alphas = None
        self.QUBO_minimisation_score = None

        self.support_ids = None
        self.support_vectors = None
        self.support_targets = None
        self.support_alphas = None

        self.bias = None

    def make_QUBO_problem(self, X_train, t_train):
        """Given X_train, t_train data to train the model. 
        Creates Q, q matrices and defines the objective function for the QUBO problem.
        """
        self.Q, self.q = qSVM.make_QUBO_matrices(X_train, t_train, self.kernel_func, self.param, self.B, self.K, self.R)
        self.ObjectiveFunction = annealing.QObjectiveFunction(self.Q, self.q)    
        return self
    
    def set_annealing_parameters(self, cooling_param, m_rate, annealing_iterations, T0):
        """
        annealing params are set as defualt, but they can be changed with this method. 
        """
        self.cooling_param = cooling_param
        self.m_rate = m_rate
        self.annealing_iterations = annealing_iterations
        self.T0 = T0
        return self
    
    def set_bias(self, bias):
        """Sets the model's bias paramter --- used for the decision function --- to the given input"""
        self.bias = bias
        return self
    
    def fit(self, X_train, t_train):
        """
        X_train, t_train: The same data which was used to create the QUBO matrices
        """
        assert self.ObjectiveFunction, "Objective function not set. Run make_QUBO_problem method"

        encoded_alphas = annealing.run_annealing(self.cooling_param, self.m_rate, self.annealing_iterations, self.T0, self.ObjectiveFunction)
        self.ObjectiveScore = self.ObjectiveFunction.evaluate(encoded_alphas)
        self.alphas = qSVM.decode(encoded_alphas, self.B, self.K)

        self.support_ids, self.support_vectors, self.support_targets, self.support_alphas = kSVM.get_support_vectors(X_train, t_train, self.alphas, self.C)
        self.bias = kSVM.discriminant_bias(X_train, t_train, self.alphas, self.C, self.kernel_func, self.param)

        return self
    
    def decision_function(self, X_test):
        return kSVM.score_kSVM(X_test, self.support_vectors, self.support_targets, self.support_alphas, self.kernel_func, self.param, self.bias)

    def predict_proba(self, X_test):
        scores = self.decision_function(X_test)
        probs = 1 / (1 + np.exp(-scores))
        return probs
    
    def predict(self, X_test):
        """Returns the predicted class for the test data"""
        return np.sign(self.decision_function(X_test))
    
def main():
    print("This file contains SoftMarginKernelClassifier and QUBOSoftMarginClassifier classes. \n")

if __name__ == "__main__":
    main()