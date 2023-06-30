import numpy as np
import matplotlib.pyplot as plt

import kernel_SVM_functions as kSVM

from classifiers import SoftMarginKernelClassifier
from sklearn.model_selection import train_test_split


def plot(X, t, model, contour_func, fig = None, ax = None, colorbar = False, support_vecs = False):
    
    """
    Takes in data X, t which will be visualised. Not necessarily the data which the model is trained on.
    contour_func: either 'predict_proba' or 'decision_func'. The function which denotes the contour plot in the background.
    colorbar: True (default) shows colorbar next to plot
    support_vecs: False (default) shows support vectors or not.     
    """
    assert contour_func == 'predict_proba' or contour_func == 'decision_function', f'{contour_func} is not valid. Must be either predict_proba or decision_function'
    if contour_func == 'predict_proba':
        cf_function = model.predict_proba
    else:
        cf_function = model.decision_function

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
    cf_values = cf_function(np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)]))
    cf_values = cf_values.reshape(xx.shape)
    #Finding the prediction boundary
    prediction = model.predict(np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)]))
    prediction = prediction.reshape(xx.shape)

    if fig:

        #Optionally pass in an axis to plot onto
        if contour_func == 'predict_proba':
            levels = np.linspace(0, 1, 20)
        else:
            #If it is the decision function instead
            min_cf, max_cf = int(min(cf_values.flatten())), int(max(cf_values.flatten())) + 1
            levels = np.linspace(min_cf, max_cf, 20)

        cf = ax.contourf(xx, yy, cf_values, cmap = 'Spectral', levels = levels, extend = 'min', alpha = 0.8)
        cs = ax.contour(xx, yy, prediction, [0], cmap = 'bone')
        ax.scatter(X[:, 0], X[:, 1], c = t, cmap = 'bwr_r', edgecolors = 'w', zorder = 3, alpha = 0.8)

        #Whether to show support vectors
        if support_vecs:
            ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], color = 'lightgrey', edgecolors = 'w', zorder = 5)

        if colorbar:
            if contour_func == 'predict_proba':
                cbar = fig.colorbar(cf, ax = ax, label='Probability')
                cbar.set_ticks([0, 0.5, 1])
            else:
                cbar = fig.colorbar(cf, ax = ax, label = 'Score')
                cbar.set_ticks([min_cf, 0, max_cf])

    else:
        #If no axis, plot a new figure.
        fig, ax = plt.subplots(figsize = (4, 4))

        if contour_func == 'predict_proba':
            levels = np.linspace(0, 1, 20)
        else:
            #If it is the decision function instead
            min_cf, max_cf = int(min(cf_values.flatten())), int(max(cf_values.flatten())) + 1
            levels = np.linspace(min_cf, max_cf, 20)

        cf = ax.contourf(xx, yy, cf_values, cmap = 'Spectral', levels = levels, extend = 'min', alpha = 0.8)
        cs = ax.contour(xx, yy, prediction, [0], cmap = 'bone')
        ax.scatter(X[:, 0], X[:, 1], c = t, cmap = 'bwr_r', edgecolors = 'w', zorder = 3, alpha = 0.8)

        #Whether to show support vectors
        if support_vecs:
            ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], color = 'lightgrey', edgecolors = 'w', zorder = 5)

        if colorbar:
            if contour_func == 'predict_proba':
                cbar = fig.colorbar(cf, ax = ax, label='Probability')
                cbar.set_ticks([0, 0.5, 1])
            else:
                cbar = fig.colorbar(cf, ax = ax, label = 'Score')
                cbar.set_ticks([min_cf, 0, max_cf])
                
        plt.show()

def main():
    print("Function does functional things.")
    print("File contains plot function to plot a model's decision surface")
    

if __name__ == "__main__":
    main()