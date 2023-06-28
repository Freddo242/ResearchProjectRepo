import numpy as np
import matplotlib.pyplot as plt

import kernel_SVM_functions as kSVM

from classifiers import SoftMarginKernelClassifier
from sklearn.model_selection import train_test_split


def plot(X, t, model, fig = None, ax = None, colorbar = False, support_vecs = False):
    
    """
    Takes in data X, t which will be visualised. Not necessarily the data which the model is trained on.
    colorbar: True (default) shows colorbar next to plot
    support_vecs: False (default) shows support vectors or not.     
    """
    res = 100
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    delta = max((x_max-x_min)/5, (y_max-y_min)/5)
    x_min -= delta
    y_min -= delta
    x_max += delta
    y_max += delta
    #All values of xx and yy to create the contour plot from.
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),np.linspace(y_min, y_max, res))
    #Finding the probability contour map
    proba = model.predict_proba(np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)]))
    proba = proba.reshape(xx.shape)
    #Finding the prediction boundary
    prediction = model.predict(np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)]))
    prediction = prediction.reshape(xx.shape)

    if fig:
        #Optionally pass in an axis to plot onto
        levels = np.linspace(0, 1, 20)
        cf = ax.contourf(xx, yy, proba, cmap = 'Spectral', levels = levels, extend = 'min', alpha = 0.8)
        cs = ax.contour(xx, yy, prediction, [0], cmap = 'bone')
        ax.scatter(X[:, 0], X[:, 1], c = t, cmap = 'bwr_r', edgecolors = 'w', zorder = 3, alpha = 0.8)
        #Whether to show support vectors
        if support_vecs:
            ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], color = 'lightgrey', edgecolors = 'w', zorder = 5)
        if colorbar:
            cbar = fig.colorbar(cf, ax = ax, label='Probability')
            cbar.set_ticks([0, 0.5, 1])

    else:
        #If no axis, plot a new figure.
        fig, ax = plt.subplots(figsize = (4, 4))

        levels = np.linspace(0, 1, 20)
        cf = ax.contourf(xx, yy, proba, cmap = 'Spectral', levels = levels, extend = 'min', alpha = 0.8)
        cs = ax.contour(xx, yy, prediction, [0], cmap = 'bone')
        ax.scatter(X[:, 0], X[:, 1], c = t, cmap = 'bwr_r', edgecolors = 'w', zorder = 3, alpha = 0.8)
        #Whether to show support vectors
        if support_vecs:
            ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], color = 'lightgrey', edgecolors = 'w', zorder = 5)
        if colorbar:
            cbar = fig.colorbar(cf, ax = ax, label='Probability')
            cbar.set_ticks([0, 0.5, 1])
        plt.show()

def main():
    print("Function does functional things.")
    print("File contains plot function to plot a model's decision surface")
    

if __name__ == "__main__":
    main()