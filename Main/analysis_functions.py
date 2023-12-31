import numpy as np
import matplotlib.pyplot as plt

def find_best(input_arr, n):
    """Finds indices of maximum n values in an array"""
    arr = input_arr.copy()
    indices = []
    arr_shape = arr.shape
    for i in range(n):
        max_index = np.argmax(arr)
        index = np.unravel_index(max_index, arr_shape)
        indices.append(index)
        arr[index] = 0

    return indices

def find_worst(input_arr, n):
    """Finds indices of maximum n values in an array"""
    arr = input_arr.copy()
    indices = []
    arr_shape = arr.shape
    for i in range(n):
        min_index = np.argmin(arr)
        index = np.unravel_index(min_index, arr_shape)
        indices.append(index)
        arr[index] = 0

    return indices

def get_params(values_list, index):
    assert len(values_list) == len(index), f"Index is not the same length as values_list. {len(values_list)} != {len(index)}"

    return [values_list[i][index[i]] for i in range(len(index))]

def plot_top_models(models, accuracys, aurocs, fig = None, ax = None, save = False):
    """
    models: tuples of parameters for each model (B, K, R, gamma)
    accuracys: (mean, std) of the acuracy for each set of parameters
    aurocs: (mean, std) of the auroc for each set of parameters

    need to add shaded areas for standard deviations for n models. Can be done with plt fill_between."""

    tick_labels = [rf"B={models[i][0]}, K={models[i][1]}, $\xi$={models[i][2]}, $\gamma$={models[i][3]}" for i in range(len(models))]

    if ax:

        ax.plot(accuracys[0], color = 'green', marker = 'D', label = 'Accuracy')
        ax.fill_between(np.arange(0, 20, 1), 
                        accuracys[0] + accuracys[1], 
                        accuracys[0] - accuracys[1], 
                        color = 'lightgreen',
                        alpha = 0.6)

        ax.plot(aurocs[0], color = 'blue', marker = 'p', label = 'AUROC')
        ax.fill_between(np.arange(0, 20, 1), 
                        aurocs[0] + aurocs[1], 
                        aurocs[0] - aurocs[1], 
                        color = 'lightblue',
                        alpha = 0.6)

        ax.set_ylim(0.49, 1.01)
        ax.set_yticks(np.arange(0.5, 1.01, 0.1), labels = [ str(round(val, 1)) for val in np.arange(0.5, 1.01, 0.1)])
        
        #ticks will be at 0 to 19
        ax.set_xlim(-1, 20)
        ax.set_xticks(np.arange(0, 20, 1), tick_labels, rotation = 'vertical')

        plt.legend()

    else:
        
        plt.figure(figsize = (6, 4))

        plt.plot(accuracys[0], color = 'green', marker = 'D', label = 'Accuracy')
        plt.fill_between(np.arange(0, 20, 1), 
                        accuracys[0] + accuracys[1], 
                        accuracys[0] - accuracys[1], 
                        color = 'lightgreen',
                        alpha = 0.6)

        plt.plot(aurocs[0], color = 'blue', marker = 'p', label = 'AUROC')
        plt.fill_between(np.arange(0, 20, 1), 
                        aurocs[0] + aurocs[1], 
                        aurocs[0] - aurocs[1], 
                        color = 'lightblue',
                        alpha = 0.6)

        plt.ylim(0.49, 1.05)
        plt.yticks(np.arange(0.5, 1.01, 0.1), labels = [ str(round(val, 1)) for val in np.arange(0.5, 1.01, 0.1)])
        
        #ticks will be at 0 to 19
        plt.xlim(-1, 20)
        plt.xticks(np.arange(0, 20, 1), tick_labels, rotation = 'vertical')

        plt.legend()
        if save:
            plt.savefig('top_models.png', dpi = 256)

        plt.show()

from matplotlib.ticker import AutoMinorLocator

def models_auc_boxplot(models, auroc, top_csvm_auroc, fig = None, ax = None):

    tick_labels = [rf"B={models[i][0]}, K={models[i][1]}, $\xi$={models[i][2]}, $\gamma$={models[i][3]}" for i in range(len(models))]

    boxprops = dict(linewidth=0.4, color='black')
    flierprops = dict(marker='.', markerfacecolor='lightblue', markersize=8, markeredgewidth = 0.4,
                  markeredgecolor='black')
    whiskerprops = dict(linewidth=1)

    low_ylim = np.floor(10 * np.min(auroc.flatten())) / 10
    print(low_ylim)

    if fig:
        
        bplot = ax.boxplot(auroc, widths = 0.6, boxprops = boxprops, whiskerprops = whiskerprops, flierprops = flierprops, patch_artist = True, zorder = 1)

        for patch in bplot['boxes']:
            patch.set_facecolor('lightblue')

        ax.plot(np.arange(1, 21), np.full(20, top_csvm_auroc), color = 'green', linewidth= 1, linestyle='--', zorder = 2, label = 'cSVM AUROC')

        #set ylim
        ax.set_ylim(low_ylim - 0.05, 1.01)
        #Set ylabels and inserting cSVM label
        ylabels = [str(round(val, 1)) for val in np.arange(low_ylim, 1.01, 0.1)]

        #Setting y ticks

        ax.set_yticks(np.arange(low_ylim, 1.01, 0.1), labels = ylabels)
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        #ticks will be at 1 to 20.
        ax.set_xlim(0, 21)
        ax.set_xticks(np.arange(1, 21, 1), tick_labels, rotation = 'vertical')

    else:
        fig, ax = plt.subplots(figsize = (4, 4))

        bplot = ax.boxplot(auroc, widths = 0.6, boxprops = boxprops, whiskerprops = whiskerprops, flierprops = flierprops, patch_artist = True, zorder = 1)

        for patch in bplot['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_ylim(0.49, 1.01)
        ylabels = [ str(round(val, 1)) for val in np.arange(0.5, 1.01, 0.1)]
        ylabels.insert(3, 'cSVM')
        ax.set_yticks(np.insert(np.arange(0.5, 1.01, 0.1), 3, top_csvm_auroc), labels = ylabels)

        #ticks will be at 0 to 19
        ax.set_xlim(0, 21)
        ax.set_xticks(np.arange(1, 21, 1), tick_labels, rotation = 'vertical')

        plt.show()

    return None


def main():
    print("File contains functions used for analysis of accuracy and auroc results from SVM.")

    models = [(2, 3, 1, 0.125)] * 20
    accuracys = np.random.uniform(0.8, 1, size = 20)
    aurocs = np.random.uniform(0.5, 1, size = 20)
    plot_top_models(models, accuracys, aurocs)


if __name__ == "__main__":
    main()