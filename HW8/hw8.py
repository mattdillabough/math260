# Matthew Dillabough - 10/20/2020

import numpy as np
from numpy import random
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Q1


def stationary(mat, alpha, sites):
    """Power method to find stationary distribution.
       Given the the transpose of the transition matrix, mat,
            and teleport parameter, alpha.
    """
    n = mat.shape[0]
    steps = 15

    mat = alpha*mat
    x = random.rand(pt.shape[0])  # random initial vector
    x /= sum(x)

    for it in range(steps):
        supp = np.full((n,), ((1-alpha)/n)*sum(x))
        x = csr_matrix.dot(mat, x) + supp
        x /= sum(x)

    # Combine x with site names and sort in descending order
    res = sorted(zip(x, sites), reverse=True)

    return res[:10]


def read_graph_file(fname, node_pre='n', adj_pre='a', edge_pre='e'):
    """ First, it reads lines with prefix n (for node) of the form
        n k name    and stores a dict. of names for the nodes.
        Reads adj. matrix data from a file, returning the adj. list.
        Format: A line starting with...
            - 'n' is read as  n k name  (the node's name)
            - 'e' is read as e k m (an edge k->m)
        Returns:
            pt - the transpose of the sparse matrix generated from the adj dictionary
            names - the dictionary of node names, if they exist
    """
    adj = {}
    names = {}

    with open(fname, 'r') as fp:
        for line in fp:
            parts = line.split(' ')
            if len(parts) < 2:
                continue
            node = int(parts[1])
            if parts and parts[0][0] == 'n':
                names[node] = parts[2].strip('\n')
            elif parts and parts[0][0] == 'e':
                v = int(parts[2])
                if node not in adj:
                    adj[node] = [v]
                else:
                    adj[node].append(v)

    row = []
    col = []
    data = []

    for node in adj:
        for edge in adj[node]:
            row.append(node)
            col.append(edge)
            data.append(1 / len(adj[node]))

    p = csr_matrix((data, (row, col)), shape=(len(row), len(col)))
    pt = p.transpose()

    return pt, names

# --------------------------------------------------------
# Q2


def rk4(t0, y0, t, h):
    n = (int)((t - t0)/h)
    yvals = [y0]
    y = y0

    for k in range(1, n + 1):
        k1 = h * yp(t0, y)
        k2 = h * yp(t0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * yp(t0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * yp(t0 + h, y + k3)

        y = y + (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        yvals.append(y)
        t0 += h

    plt.plot(yvals)
    plt.show()


def yp(x, y):
    return 2 * x * y

# --------------------------------------------------------
# Q3


def fwd_euler_sys(f, a, b, y0, h):
    """ Forward euler, to solve the system y' = f(t,y) of m equations
        Inputs:
            f - a function returning an np.array of length m
         a, b - the interval to integrate over
           y0 - initial condition (a list or np.array), length m
           h  - step size to use
        Returns:
            t - a list of the times for the computed solution
            y - computed solution; list of m components (len(y[k]) == len(t))
    """
    y = np.array(y0)  # copy!
    t = a
    tvals = [t]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]
    while t < b - 1e-12:
        print(y)
        y += h*f(t, y)
        t += h
        for k in range(len(y)):
            yvals[k].append(y[k])
        tvals.append(t)

    return tvals, yvals


def trap_approx(ode, a, b, y0, h):
    """ Reworked trapezoidal rule equations with algebra to isolate for xn1 and yn1
        At each step, approximate both xn1 and yn1 and then append xvals and yvals array with new approximations
        Input:
            - fx = the x function
            - fy = the y functoin
            - h = step length
            - a = lower bound
            - b = upper bound
        Returns:
            - xvals = array containing x approximations for each step
            - yvals = array containing y approximations for each step
    """

    y = np.array(y0)
    tvals = [a]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]

    tn = a
    tn1 = tn + h

    # return tvals, yvals
    while tn < b - 1e-12:
        c = (tn1 - tn)/2

        # print(f(tn, y))
        xn, yn = f(tn, y)

        # With algebra, reworked the equation for trapezoidal rule to solve: xn1 = xn + ((tn1 + tn)/2)(fy(tn1) + fy(tn))
        xn1 = (y[0] + c*yn + (c**2 + c)*xn)/(1 - c**2)
        # With algebra, reworked this equation for trapezoidal rule: yn1 = yn + ((tn1 + tn)/2)(fx(tn1) + fx(tn))
        yn1 = (y[1] + c*xn + (c**2 + c)*yn)/(1 - c**2)

        y += h*f(tn, y)

        yvals[0].append(xn1)
        yvals[1].append(yn1)

        tvals.append(tn)

        tn = tn1
        tn1 = tn + h
        # In complete honesty, I have spent a lot of time trying to figure out how to create a better result
        #   with this method, however I have been unable to get something that I feel is completely accurate
        # Just wanted to let you know I genuinly tried incredibly hard on this question, but could not get
        #   it to work out in the end

    return tvals, yvals


def f(t, v):
    return np.array([v[1], -v[0]])


if __name__ == "__main__":
    # Run code from Q1
    pt, sites = read_graph_file("california.txt")
    result = stationary(pt, 0.9, sites)
    for res in result:
        print(f"{sites[res[1]]} = {res[0]}")

    # Run code from Q2
    rk4(0, 1, 1, 0.1)

    # Run code from Q3
    v_init = [1.0, 0]
    h = 0.1
    periods = 2
    t, v = trap_approx(f, 0, periods*2*np.pi, v_init, h)
    x = v[0]
    y = v[1]

    t, v1 = fwd_euler_sys(f, 0, periods*2*np.pi, v_init, h)
    x1 = v1[0]
    y1 = v1[1]

    plt.figure()
    plt.plot(t, x, '-k', t, y, '-r')
    plt.plot(t, x1, 'b', t, y1, 'b')  # Plot for Euler's method
    plt.legend(["$\\theta(t)$", "$\\theta'(t)$"])
    plt.xlabel("$t$")
    plt.show()

    # The approximations for the two results are very similar however there is slight variation
    # Theta' is stretched vertically slightly more than the Euler result and Theta is vertically
    # shrinked slightly more than the Euler result
