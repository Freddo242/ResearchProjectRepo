import numpy as np
import matplotlib.pyplot as plt
import sys

class QObjectiveFunction:

    def __init__(self, Q, q):
        self.Q = Q
        self.q = q
        self.D = Q.shape[0]

    def evaluate(self, s):
        return ((s.T @ self.Q) @ s + s.T @ self.q).flatten()[0]

def max_cut_problem(s):
    """standard QUBO problem for training purposes"""
    Q = -np.array([[2, -1, -1, 0, 0],
              [-1, 2, 0, -1, 0], 
              [-1, 0, 3, -1, -1], 
              [0, -1, -1, 3, -1],
              [0, 0, -1, -1, 2]])
    q = np.zeros(shape = (5, 1))

    assert s.shape == (5, 1), f"candidate has incorrect shape {s.shape}, expecting {(5, 1)}"
    return ((s.T @ Q) @ s + s.T @ q).flatten()[0]

def get_neighbour(s, D, m_rate):
    """returns a new suggestion for the next solution"""
    flip_probs = np.random.uniform(0, 1, D)
    #creating a mask where 1 is if the bit is flipped and 0 is if it is not.
    flip_mask = np.zeros(D)
    flip_mask[flip_probs < m_rate] = 1
    #flipping values in s where flip mask is 1
    new_s = s.copy()
    new_s[flip_mask == 1] = abs(new_s[flip_mask == 1] - 1)
    return new_s

def acceptance_probability(e_old, e_new, T):
    """decides the probability with which to accept the new solution depending on whether it is better or worse than the current."""
    if e_new < e_old:
        return 1
    else:
        diff = e_new - e_old
        return np.exp(-diff / T)

def cool_T(t0, k, alpha):
    """Updates the temperature parameter each iteration"""
    return t0 / (1 + alpha * np.log(1 + k))

def temperature_probability_graph(t0, alpha):
    """Produces a graph showing the trend of temperature and probability over time"""
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    N = 100
    xs = np.arange(0, N)
    ts = np.array([cool_T(t0, k, alpha) for k in range(N)])
    ps = np.array([acceptance_probability(-1, 0, ts[i]) for i in range(N)])

    ax[0].set_title('Temperature')
    ax[0].plot(xs, ts, color='purple')

    ax[1].set_title('Probability')
    ax[1].plot(xs, ps, color='red')

    plt.show()

def run_annealing(alpha, m_rate, N, t0, energy_func, graph = False, print_round = False):
    #Assuming energy_func is an instance of the QObjectivefunction class

    D = energy_func.D
    s = np.random.randint(0, 2, size = (D, 1))
    energy_function = energy_func.evaluate
    E = energy_function(s)
    T = t0

    energy_record = [E]

    for k in range(N):
        s_new = get_neighbour(s, D, m_rate)
        e_new = energy_function(s_new)
        #if e_new < E:
        #    print("new is better.")
        #else:
        #    print("Diff ", e_new - E)
        p_acc = acceptance_probability(E, e_new, T)
        if np.random.uniform(0, 1) < p_acc:
            s = s_new.copy()
            E = e_new.copy()
            energy_record.append(E)
        
        T = cool_T(t0, k, alpha)
        
        if print_round:
            print("Round ", k)
            print("T ", T)
            print("Energy, ", E)
            print("p, ", p_acc)
            print("-----------")
    
    if graph:
        temperature_probability_graph(t0, alpha)

    return s

def main():

    alpha = 4
    m_rate = 0.2
    N = 100
    t0 = 10
    energy_function = max_cut_problem
    D = 5
    s = np.random.randint(0, 2, size = (D, 1))
    E = energy_function(s)
    T = t0

    energy_record = []
    probability_record = []

    for k in range(N):
        s_new = get_neighbour(s, D, m_rate)
        e_new = energy_function(s_new)
        #if e_new < E:
        #    #print("new is better.")
        #    continue
        #else:
        #    diff = e_new - E
        #    print(diff)
        #    probability_record.append(np.exp(-diff / T))

        p_acc = acceptance_probability(E, e_new, T)
        if np.random.uniform(0, 1) < p_acc:
            s = s_new.copy()
            E = e_new.copy()
            energy_record.append(E)
        
        T = cool_T(t0, k, alpha)
        
        print("Round ", k)
        print("T ", T)
        print("Energy, ", E)
        print("p, ", p_acc)
        print("-----------")

    print(probability_record, '\n')
    print(energy_record)

    temperature_probability_graph(t0, alpha)


if __name__ == "__main__":
    main()


