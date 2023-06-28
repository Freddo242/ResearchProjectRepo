import numpy as np
from scipy.stats import norm as norm
import matplotlib.pyplot as plt

np.random.seed(seed = 11235813)

sigmas = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4]
r_pos = 1
r_neg = 0.15
n = 60

data_matrices = []

for sigma in sigmas:

    phi = np.linspace(0, 4 * np.pi, 2 * n, endpoint = False)

    unit_circle_points = np.array([np.cos(phi), np.sin(phi)]).T
    r = np.hstack((np.full(n, r_pos), np.full(n, r_neg))).reshape(-1, 1)
    s = norm.rvs(0, sigma, size = (2 * n, 2))

    t = np.hstack((np.full(n, 1), np.full(n, -1))).reshape(-1, 1)

    X = r * unit_circle_points + s
    X = np.hstack((X, t))
    data_matrices.append(X)

for i, s in enumerate(sigmas):
    with open(f'synth_data/synth_{str(s)}.csv', 'w') as f:
        np.savetxt(f, data_matrices[i], delimiter=',', newline='\n')

for i, X in enumerate(data_matrices):
    plt.figure(figsize=(4, 4))
    row = int(i / 2)
    col = i % 2

    plt.scatter(X[: n, 0], X[: n, 1], color = 'blue', s = 5)
    plt.scatter(X[n: , 0], X[n: , 1], color = 'orange', s = 5)
    plt.axis('off')
    plt.title(str(sigmas[i]))
    plt.savefig(f'synth_data/{str(sigmas[i])}_fig.png', dpi=256)
    plt.clf()
    
