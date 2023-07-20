import numpy as np

def rbf_kernel(x, y, param):
    """
    calculates the gaussian kernel transformation between two vectors x and y. 
    x, y: vectors should have the same shape (d, ) where d is the dimension of the problem.
    param: a 
    """
    diff = x - y
    K = np.exp(-param * (diff.T @ diff))
    return K

def train_qp(P ,q, G, h, A, b):
    """
    returns the solution from cvxopt qp solver
    """
    from cvxopt import solvers

    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P ,q, G, h, A, b)
    except ValueError:
        #Handling if cvxopt cannot solve the problem.
        return None
    
    return np.array(sol['x'])

def kSVM_matrices(X, t, C, kernel_func, gamma):
    """
    returns kernel SVM matrices P, q, G, h, A, b, for use in cvxopt solvers.qp() function. 
    """
    from cvxopt import matrix

    N, d = X.shape
    assert t.shape == (N, 1), f"t is required to be shape ({X.shape[0]}, 1), but is shape {t.shape}"

    P = matrix(np.array([[kernel_func(X[i], X[j], gamma) * t[i][0] * t[j][0] for i in range(N)] for j in range(N)]), tc='d')
    q = matrix(np.full(N, -1), tc='d')
    G = matrix(np.vstack((np.diag(-1 * np.ones(N)), np.diag(np.ones(N)))), tc='d')
    h = matrix(np.vstack((np.zeros((N, 1)), C * np.ones((N, 1)))), tc='d')
    A = matrix(t.reshape(1, -1), tc='d')
    b = matrix(np.array([0]), tc='d')

    return P, q, G, h, A, b

def discriminant_bias(X_train, t_train, alphas, C, kernel_func = None, gamma = None):
    """
    to calculate the 'bias' for the discriminant, linear or non-linear. 
    X_train: data matrix shape (n, d) (d dimensional problem)
    t_train: array of targets. shape (n, 1)
    alphas: result of lagrangian dual problem. shape (n, 1)
    C: regularisation parameter used. Real number.
    kernel_func: function used to calculate the kernel transformation for two vectors (x, y). If not given, we assume that 
    no kernel is used. This is the same as the dot product between the two vectors x.T * x.
    """

    N, d = X_train.shape
    assert t_train.shape == (N, 1), f"t has shape {t_train.shape}, but should have shape ({N}, 1)"
    assert alphas.shape == (N, 1), f"alphas has shape {alphas.shape}, but should have shape ({N}, 1)"

    denominator = np.dot(alphas.reshape(1, -1), (C - alphas))
    #if this is zero, the support vectors do not lie on the margin, therefore we give a bias of 0.
    if denominator[0][0] == 0:
        return 0

    if kernel_func:
        H = np.array([[kernel_func(X_train[m], X_train[n], gamma) for m in range(N)] for n in range(N)])
    else:
        H = X_train @ X_train.T

    v = t_train - H @ (alphas * t_train)
    numerator = np.dot(alphas.reshape(1, -1), (C - alphas) * v)

    return numerator[0][0] / denominator[0][0]

def get_support_vectors(X_train, t_train, alphas, C):
    #Returns support ids, support vectors, support targets, and support alphas for use in calculating score.
    N = X_train.shape[0]
    assert (t_train.shape[0] == N) and (alphas.shape[0] == N), "either t_train or alphas does not match the shape of X_train"
    
    #Find support vectors and their alphas and corresponding targets
    support_ids = np.where((alphas.flatten() > 1e-4))[0]

    support_vectors = X_train[support_ids]
    support_targets = t_train[support_ids]
    support_alphas = alphas[support_ids]

    return support_ids, support_vectors, support_targets, support_alphas

def score_kSVM(X_test, support_vectors, support_targets, support_alphas, kernel_func, param, bias):
    """alphas: list of the alpha values corresponding to each vector in X_train
    t_train: targets of the vectors in X_train
    C: maximum value an alpha can be corresponding to sum(B**i for i in range(K)) in the encoding
    kernel_func and param: to calculate the kernel function of two vectors
    bias: calculated using discriminant_bias() in SVM_functions.py
    """
    H = np.array([[support_alphas[n][0] * support_targets[n][0] * kernel_func(support_vectors[n], X_test[m], param) for n in range(len(support_vectors))] for m in range(len(X_test))])
    y = np.sum(H, axis = 1) + bias
    return y

def main():
    print("This file contians functions to assist with kernel soft margin classifiers. \n")
    print("""The following funcitons are available: \n 
    - score_kSVM \n - get_support_vectors \n - discriminant_bias \n - kSVM_matrices \n - train_qp \n - rbf_kernel \n""")
    pass

if __name__ == '__main__':
    main()
