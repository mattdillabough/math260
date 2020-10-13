# Lecture code: bee example.
# The power method for computing a stationary distribution,
# plus code for choosing a move from a state to the next.
#
# Also: pagerank example from class (with four nodes)

import numpy as np
from numpy.random import uniform, rand   # for a random number


# --------------------------------------------------------
# Stationary distribution calculation from theory

def stationary(pt, tol):
    """Power method to find stationary distribution.
       Given the largest eigenvalue is 1, finds the eigenvector.
    """
    x = rand(pt.shape[0])  # random initial vector
    x /= sum(x)
    x1 = np.dot(pt, x)
    x1 /= sum(x1)
    stepCount = 0

    while x1 - x > tol and stepCount <= 100:
        x = x1
        x1 = np.dot(pt, x)
        x1 /= sum(x1)

        stepCount += 1

    return x1


# --------------------------------------------------------
# Pagerank example from lecture

def pagerank_small():
    """ small pagerank example from lecture (four nodes) """
    n = 4
    p_matrix = np.array([[0, 1/2, 0, 1/2],
                         [0, 0, 1, 0],
                         [1/3, 1/3, 0, 1/3],
                         [0, 0, 1, 0]])

    # not efficient here; just to illustrate calculation
    alpha = 0.95
    pt_mod = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            pt_mod[j, k] = alpha*p_matrix[k, j] + (1-alpha)*(1/n)

    # NOTE: you can shorthand this with
    # pt_mod = alpha*p_matrix.transpose() + (1-alpha)/n
    # which uses numpy's rules for adding scalars to arrays
    # For efficient code, you'd construct the transpose directly
    tol = 10
    dist = stationary(pt_mod, tol)
    print(dist)


if __name__ == "__main__":
    pagerank_small()
