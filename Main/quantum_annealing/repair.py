import sys
sys.path.append('../')
import copy
import pickle

import numpy as np
np.random.seed(314159)
import pandas as pd

import QUBO_SVM_functions as qSVM
from sklearn.model_selection import train_test_split
from quantum_classifier import QSVMq
from quantum_annealing_functions import stratified_sample
from kernel_SVM_functions import rbf_kernel



def main():

    #Get data from file
    datafile = 'synth_0.4.csv'
    datafilepath = f'../synth_data/{datafile}'
    train_size = 0.4

    data = np.loadtxt(datafilepath, delimiter = ',')
    X = data[:, :-1]
    t = data[:, -1]

    #split data into train test
    X_train, X_test, t_train, t_test = train_test_split(X, t, train_size = train_size, shuffle = True, stratify = t)
    t_train = t_train.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    #Model hyper-parameters
    B = 2
    K = 2
    R = 1
    gamma = 4
    kernel_func = rbf_kernel

    QAClassifier = QSVMq(B, K, R, kernel_func, gamma)

    #Bagging paramters
    bag_size = 10
    sample_size = 18

    models = []

    for n in range(20):

        print('model ', n, '\n')

        bag = []

        for b in range(bag_size):
            print('bag: ', b)
            f_name = f'../QA_bagged_results/model-{n}_bag-{b}'

            #Taking a sample from training data
            sample_index = stratified_sample(t_train, sample_size)
            X_sample, t_sample = X_train[sample_index], t_train[sample_index]
            print(X_sample.shape, t_sample.shape)

            print('data sampled')
            #Copying classifier and making the QUBO problem using the sampled data
            clf = copy.deepcopy(QAClassifier)
            clf = clf.make_QUBO_problem(X_sample, t_sample)

            #Reading results from Annealer
            df = pd.read_csv(f_name)
            print('df read')

            #Finding alphas with lowest energy
            encoded_alphas = np.array(list(df.iloc[np.argmin(df['energy'])][1: -3]))
            alphas = qSVM.decode(encoded_alphas, B, K).reshape(-1, 1)
            print('alphas found')
            print(alphas.shape)
            #set the model with best alphas
            clf = clf.set_model(X_sample, t_sample, alphas)
            print('model set')
            bag.append(clf)

        models.append(bag)

    with open(f'../QA_bagged_results/synth-4_bagged_models', 'wb') as f:
        pickle.dump(models, f)
    print("models dumped")

if __name__ == "__main__":
    main()