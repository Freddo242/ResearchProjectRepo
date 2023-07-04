import numpy as np

from sklearn.model_selection import StratifiedKFold
from metric_functions import compute_auc_from_scores, compute_accuracy

def qp_cross_validate(X_train, t_train, k_folds, model, num_models = 1):
    """Function performs k-fold cross validation on the given model. Each fold is chosen via stratified sampling.
    Each fold is evaluated using accuracy and AUROC metrics.
    Takes in X_train, t_train data to train model.
    k_folds: number of folds to divide X_train into
    classifier: Assumed that the classifier will be either SoftMarginKernelClassifier or QUBOSoftMarginClassifier from the classifier.py file.
    num_models: the number of models to create for each fold. metrics are averaged across the models.
    Returns accuracy_results, auroc_results: the accuracy and auroc for each fold. 
    """
    
    skf = StratifiedKFold(n_splits = k_folds)

    accuracy_results = []
    auroc_results = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, t_train)):
        #Creating train and validation sets from the StratifiedKFold class
        X_train_split = X_train[train_idx]
        t_train_split = t_train[train_idx].reshape(-1, 1)
        X_test_split = X_train[test_idx]
        t_test_split = t_train[test_idx].reshape(-1, 1)

        fold_accuracy_results = []
        fold_auroc_results = []

        for i in range(num_models):
            #Split model training up since QUBO classifier requires a different 
            if model.__class__.__name__ == "QUBOSoftMarginClassifier":
                no_support_vectors = True
                #In the rate instance there are no support vectors from QUBOSVM, simply discard the model.
                while no_support_vectors:

                    model = model.make_QUBO_problem(X_train_split, t_train_split).fit(X_train_split, t_train_split)

                    if len(model.support_ids) != 0:
                        no_support_vectors = False

            else:
                model = model.fit(X_train_split, t_train_split)
                #Returning None if the problem raises an error from cvxopt 
                if model.alphas is None:
                    return None, None
                
                if len(model.support_ids) != 0:
                    no_support_vectors = False

            scores = model.decision_function(X_test_split)
            preds = model.predict(X_test_split)

            accuracy = compute_accuracy(preds, t_test_split)
            auroc = compute_auc_from_scores(scores, t_test_split)

            fold_accuracy_results.append(accuracy)
            fold_auroc_results.append(auroc)

        accuracy_results.append(np.mean(fold_accuracy_results))
        auroc_results.append(np.mean(fold_auroc_results))

    return accuracy_results, auroc_results
    

def main():
    print("file contains qp_cross_validate to perform k_fold cross validation on a given model.")


if __name__ == "__main__":
    main()