""" Assignment_1 of Neural Networks and Computational Intelligence:
 Implementation of Rosenblatt Perceptron Algorithm
"""

import numpy as np

#### Define parameters for experiments
N = [20, 40]  # number of features for each datapoint
alpha = [x / 100 for x in range(75, 325, 25)]  # ratio of (datapoint_amount / feature_amount)
n_D = 50  # number of datasets required for each value of P
n_max = 100  # maximum number of epochs

#### Params for data generation
MU = 0
SIGMA = 1


# TODO: implement gaussian data generation (done?)
def generate_data(n: int, p: int) -> [np.ndarray, np.ndarray]:
    """ Generation of artificial dataset containing P randomly generated N-dimensional feature vectors and labels.
        - The datapoints are sampled from a Gaussian distribution with mu=0 and std=1.
        - The labels are independent random numbers y = {-1,1}.
    :return: Generated dataset as a PxN numpy array; labels as Px1 numpy array.
    """
    X = []
    for i in range(n):
        X.append(np.random.normal(MU, SIGMA, p))
    X = np.asarray(X).transpose()

    y = np.asarray([np.random.choice([-1, 1]) for _ in range(len(X))])
    return X, y


# TODO: implement perceptron algorithm
def perceptron_algorithm():
    """ Implementation of the Rosenblatt Perceptron algorithm.
    :return: ...
    """
    return


# TODO: implement training logic
def train(n: int, p: int, epochs: int, data: np.ndarray):
    """ Implementation of sequential perceptron training by cyclic representation of the P examples.
     :param n: Number of features. A single chosen value from N. E.g.: N[0].
     :param p: Number of examples. A single chosen value from P. E.g.: P[0][0]. First index has to match index of N.
     :param epochs: Number of epochs.
     :param data: Dataset containing generated examples using 'generate_data' funct.
     :return: ...
     """
    w = np.array([0])
    for i in range(epochs):
        for j in range(p):
            return


# CREATING DATASETS
datasets = {20: {}, 40: {}}  # datasets with 20 and 40 features
for n in N:
    P = [int(a * n) for a in alpha]  # number of datapoints per dataset FOR the current N
    for p in P:
        datasets[n][p] = []
        for _ in range(n_D):  # create n_D datasets for each P
            datasets[n][p].append(generate_data(n, p))

# TODO: main code for stitching functions together

# Note for Alex:
# ---------------------------------------------------------------------------------------
# - Interpretation for 'datasets' data structure:
# -- first key is the number of features, the keys inside those are the number of datapoints within each dataset, e.g.:
# -- datasets[20][35] means you are accessing a dataset which has 25 features and 30 datapoints
# -- The actual array which contains the values for each datapoint in the previous example can be selected via
#       datasets[20][35][0][0], the labels array can be selected via datasets[20][35][0][1]. This is because there are
#       n_D = 50 datasets for each configuration, thus len(datasets[20][35)) would output 50 for example.
# ---------------------------------------------------------------------------------------


# TODO: Extensions
# 1) Observe behaviour of Q_l.s.
# 2) Determine embedding strengths x^mu
# 3) Use non-zero value for 'c'
# 4) Inhomogeneous perceptron with clamped inputs
# 5) ...
