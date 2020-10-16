import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def power_method(a, steps, sol):
    """ Computes the error for the power method at each step and prints the error
        Args:
            a - the (n x n) matrix
            steps - the number of iterations to take
            sol - the solution eigenvector
        Returns:
            Prints the error at each step of the power method calculation
    """
    n = a.shape[0]
    error = [0]*steps
    x = random.rand(n)
    it = 0

    while it < steps:
        q = np.dot(a, x)  # Compute a*x
        x = q/np.sqrt(q.dot(q))  # Normalize x to a unit vector

        err = x - sol  # Find difference between solution and computed vector
        err = np.square(err)  # Square all elements in error vector
        err = np.sqrt(sum(err))  # Square root of sum of error vector
        error[it] = err

        it += 1

    return error


if __name__ == "__main__":
    a = np.array([[3, 1], [0, 2]])
    sol = [1, 0]
    error = power_method(a, 100, sol)
    x = np.linspace(0, 100)

    plt.semilogy(error)
    plt.semilogy(0.82**x)
    plt.ylabel("Error")
    plt.xlabel("Step")
    plt.show()

    # Error ~= Cr^n where r is roughly 0.82
