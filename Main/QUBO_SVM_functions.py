import numpy as np
import simulated_annealing as annealing

def encode_binary(x, K):
    #Only works for B == 2 so far. i.e. binary
    binarised = np.zeros((len(x), 3))

    for i, num in enumerate(x):
        b = np.array(list(f'{num[0]:0{K}b}'), dtype=int)[::-1]
        binarised[i] = b

    binarised = binarised.flatten().reshape(-1, 1)
    return binarised

def decode(encoded_alphas, B, K):
    N = int(encoded_alphas.shape[0] / K)
    #encoded alphas is a vector of N, K length bit vectors.

    b = np.array([B ** i for i in range(K)])
    P = np.zeros((N, N * K))
    for i in range(N):
        P[i, i * K: (i + 1) * K] = b
    
    alphas = P @ encoded_alphas
    return alphas

def make_QUBO_matrices(X, t, kernel_func, param, B, K ,R):
    """
    Function builds kernel SVM QUBO matrices.
    B, K: dictate encoding. Base B with K bits
    R: penalty term to make it unconstrained
    """
    N = X.shape[0]

    H = np.array([[kernel_func(X[n], X[m], param) + R for n in range(N)] for m in range(N)])
    H = H * t
    H = H * t.reshape(1, -1)

    b = np.array([B**k for k in range(K)])
    P = np.zeros((N, K * N))
    for i in range(N):
        P[i, i*K: (i + 1) * K] = b

    q = -P.T @ np.ones((N, 1))
    Q = (P.T @ H) @ P

    return Q, q

def solve_QUBO(Q, q, m_rate, T0, cooling_param, annealing_iterations, num_solutions):
    """Takes in QUBO Matrices and simulated annealing params and returns num_solutions solutions 
    running the annealing process on the QUBO matrix"""
    #Create an objective function 
    obj_function = annealing.QObjectiveFunction(Q, q)
    #Find solution using simulated annealing
    encoded_solutions = {}
    for i in range(num_solutions):

        encoded_alphas = annealing.run_annealing(cooling_param, m_rate, annealing_iterations, T0, obj_function, graph = False)
        score = obj_function.evaluate(encoded_alphas)
        encoded_solutions[f's_{i}'] = {'score': score, 'alphas': encoded_alphas}

    return encoded_solutions

def main():
    print("""This file contains functions necessary to run a QUBO kernel Soft Margin Classifier \n 
    The functions available are: \n
    - encode_binary \n - decode \n - make_QUBO_matrices \n - solve_QUBO \n
    """)

if __name__=='__main__':
    main()

