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

def get_params(values_list, index):
    assert len(values_list) == len(index), f"Index is not the same length as values_list. {len(values_list)} != {len(index)}"

    return [values_list[i][index[i]] for i in range(len(index))]

def plot_top_models(models, accuracys, aurocs, fig = None, ax = None, save = False):
    """need to add shaded areas for standard deviations for n models. Can be done with plt fill_between."""

    tick_labels = [rf"B={models[i][0]}, K={models[i][1]}, $\xi$={models[i][2]}, $\gamma$={models[i][3]}" for i in range(len(models))]

    if ax:

        ax.plot(accuracys, color = 'green', marker = 'D')
        ax.plot(aurocs, color = 'blue', marker = 'p')

        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1, 0.2), labels = [ str(round(val, 1)) for val in np.arange(0, 1, 0.2)])
        
        #ticks will be at 0 to 19
        ax.set_xlim(-1, 20)
        ax.set_xticks(np.arange(0, 20, 1), tick_labels, rotation = 'vertical')

        plt.legend()

    else:
        
        plt.figure(figsize = (6, 4))

        plt.plot(accuracys, color = 'green', marker = 'D', label = 'Accuracy')
        plt.plot(aurocs, color = 'blue', marker = 'p', label = 'AUROC')

        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1, 0.1), labels = [ str(round(val, 1)) for val in np.arange(0, 1, 0.1)])
        
        #ticks will be at 0 to 19
        plt.xlim(-1, 20)
        plt.xticks(np.arange(0, 20, 1), tick_labels, rotation = 'vertical')

        plt.legend()
        if save:
            plt.savefig('top_models.png', dpi = 256)

        plt.show()


def main():
    print("File contains functions used for analysis of accuracy and auroc results from SVM.")

    models = [(2, 3, 1, 0.125)] * 20
    accuracys = np.random.uniform(0.8, 1, size = 20)
    aurocs = np.random.uniform(0.5, 1, size = 20)
    plot_top_models(models, accuracys, aurocs)


if __name__ == "__main__":
    main()