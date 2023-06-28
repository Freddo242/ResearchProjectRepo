import numpy as np
from cvxopt import solvers, matrices

def linear_discr_w(X_train, t_train, alphas):
    """
    Calculates the gradient vector w for the linear discriminant.
    Used for linear SVM
    """
    #Assertions to ensure that t and alphas are the correct shapes
    N, d = X_train.shape
    assert t_train.shape == (N, 1), f"t has shape {t_train.shape}, but should have shape ({N}, 1)"
    assert alphas.shape == (N, 1), f"alphas has shape {alphas.shape}, but should have shape ({N}, 1)"

    alpha_t = alphas * t_train
    w = np.sum(X_train * alpha_t, axis = 0)
    return w

def get_linear_discriminant(X_train, t_train, alphas, C):
    """
    returns both the gradient of the linear discriminant and the bias where
    x.T @ W + b is the linear discriminant
    """
    N, d = X_train.shape
    assert t_train.shape == (N, 1), f"t has shape {t_train.shape}, but should have shape ({N}, 1)"
    assert alphas.shape == (N, 1), f"alphas has shape {alphas.shape}, but should have shape ({N}, 1)"

    w = linear_discr_w(X_train, t_train, alphas)
    b = discriminant_bias(X_train, t_train, alphas, C)

    return w, b

def main():
    pass

if __name__ == "__main__":
    main()