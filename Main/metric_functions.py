import numpy as np
import matplotlib.pyplot as plt

def compute_accuracy(preds, targets):
    return np.sum((preds.flatten() - targets.flatten() == 0)) / len(preds)

def true_positive_rate(preds, targets):
    idx = np.argsort(-np.array(preds))
    sorted_targets = np.array(targets)[idx]
    TPR = np.cumsum(sorted_targets == 1) / np.sum(np.array(targets) == 1)
    return TPR

def false_positive_rate(preds, targets):
    idx = np.argsort(-np.array(preds))
    sorted_targets = np.array(targets)[idx]
    FPR = np.cumsum(sorted_targets != 1) / np.sum(np.array(targets) != 1)
    return FPR

def compute_auc_from_scores(preds, targets):
    TPR = true_positive_rate(preds, targets)
    FPR = false_positive_rate(preds, targets)
    return compute_auc(TPR, FPR)

def compute_auc(TPR, FPR):
    #Computes AUROC using the midpoint rule. i.e. splitting the area into rectangles.
    areas = [(FPR[i+1] - FPR[i])*TPR[i] for i in range(len(TPR)-1)]
    auc = np.sum(areas)
    return auc

def compute_tpr_fpr_range(scores_list, y_test, low_quantile, high_quantile):
    """
    Returns: tpr_low, tpr_high, tpr_mid, unique_fpr 
    for the distribution of TPRs given each FPR
    """
    scores = scores_list.flatten()
    targets = y_test.flatten()

    TPRs = true_positive_rate(scores, targets)
    FPRs = false_positive_rate(scores, targets)
    
    tpr_mid = []
    tpr_high = []
    tpr_low = []
    unique_fpr = np.unique(FPRs)

    for ufp in unique_fpr:
        tpr_distr = sorted(TPRs[FPRs == ufp])
        tpr_mid.append(np.median(tpr_distr))
        tpr_low.append(tpr_distr[int(len(tpr_distr) * low_quantile)])
        tpr_high.append(tpr_distr[int(len(tpr_distr) * high_quantile)])

    return tpr_low, tpr_high, tpr_mid, unique_fpr    

def confusion_table(targets, preds):
    
    def f(i, j):
        return np.sum(np.logical_and(targets == j, preds == i))
    
    n = len(set(targets))
    c_table = np.array([[f(i, j) for i in range(n)] for j in range(n)])
    
    return c_table

def plot_roc(tpr_low, tpr_high, tpr_mid, fpr):
    auc = compute_auc(tpr_mid, fpr)
    
    plt.figure(figsize = (4, 4))
    
    plt.plot(fpr, tpr_mid, color = 'black', linewidth = 1, zorder = 2)
    plt.scatter(fpr, tpr_mid, color = 'black', marker = 'o', s = 15, zorder = 3)
    plt.fill_between(fpr, tpr_low, tpr_high, color = '#C0C0C0', zorder = 0)

    plt.grid(visible = True)
    plt.title(f'AUC ROC:{round(auc, 2)}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

def main():
    print("This does nothing... \n")
    print("Except print that \n")
    print("And these last few messages explaining the initial printing \n")
    print("But this is the last! ")

if __name__ == "__main__":
    main()