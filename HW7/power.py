import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def power_method(a, steps, sol):
    """ simple implementation of the power method, using a fixed
        number of steps. Computes the error at each step and prints the error
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

    while it < steps:  # other stopping conditions would go here
        q = np.dot(a, x)  # compute a*x
        x = q/np.sqrt(q.dot(q))  # normalize x to a unit vector

        err = x - sol  # Find difference between solution and computed vector
        err = np.square(err)  # Square all elements in error vector
        err = np.sqrt(sum(err))  # Square root of sum of error vector
        error[it] = err

        it += 1

    return error


if __name__ == "__main__":   # example from lecture (2x2 matrix)
    a = np.array([[3, 1], [0, 2]])
    sol = [1, 0]
    error = power_method(a, 100, sol)

    plt.plot(error)
    plt.ylabel("Error")
    plt.xlabel("Step")
    plt.show()
