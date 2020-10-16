# Lecture code: bee example.
# The power method for computing a stationary distribution,
# plus code for choosing a move from a state to the next.
#
# Also: pagerank example from class (with four nodes)

import numpy as np
from numpy.random import uniform, rand   # for a random number


# --------------------------------------------------------
# Stationary distribution calculation from theory

def stationary(pt):
    """Power method to find stationary distribution.
       Given the largest eigenvalue is 1, finds the eigenvector
       within a max number of steps and within a certain tolerance.
    """
    x = rand(pt.shape[0])  # random initial vector
    x /= sum(x)
    x1 = np.dot(pt, x)
    x1 /= sum(x1)
    stepCount = 0
    diff = computeDiff(x, x1)

    while diff > 0.001 and stepCount <= 10:
        x[:] = x1
        x1 = np.dot(pt, x)
        x1 /= sum(x1)
        diff = computeDiff(x, x1)
        stepCount += 1

    return x1


def computeDiff(x, x1):
    """Returns the square root of the sum of squares for the difference of two vectors"""
    diff = x1 - x
    diff = np.square(diff)
    diff = np.sqrt(sum(diff))

    return diff

# --------------------------------------------------------
# Pagerank example from lecture


def pagerank_small():
    n = 5
    p_matrix = np.array([[0, 1/3, 1/3, 1/3, 0],
                         [1/2, 0, 0, 0, 1/2],
                         [1/2, 1/2, 0, 0, 0],
                         [1/2, 1/2, 0, 0, 0],
                         [0, 1, 0, 0, 0]])

    alpha = 0.95
    pt_mod = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            pt_mod[j, k] = alpha*p_matrix[k, j] + (1-alpha)*(1/n)

    dist = stationary(pt_mod)
    print(dist)


if __name__ == "__main__":
    pagerank_small()
    # Result begins to converge after only 10 steps with alpha = 0.95
    # Returns [0.27224289 0.35742411 0.09507099 0.09507099 0.18019102]

    # There is no such alpha value which will change the highest ranked page
